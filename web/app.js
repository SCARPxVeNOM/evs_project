/* ==========================================================================
   Rainwater Harvesting Advisor - browser client

   Vanilla JS, no build step. Talks to the FastAPI service on the same origin:
   /api/parameters and /api/hoblis to build the form, /api/location for the site
   card, /api/model for the classifier panel, /api/recommend for the report.

   Leaflet is loaded from a CDN. If it is unavailable the map hides itself and
   the hobli dropdown takes over, so the tool still works offline.
   ========================================================================== */

(function () {
  "use strict";

  const state = {
    lat: null,
    lon: null,
    params: null,
    hoblis: [],
    model: null,
    map: null,
    marker: null,
    lastReport: null,
  };

  const $ = (id) => document.getElementById(id);

  /* ----------------------------------------------------------- formatting */

  const num = (v, digits = 0) =>
    Number(v).toLocaleString("en-IN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });

  const kl = (litres) => num(litres / 1000, litres < 10000 ? 1 : 0);
  const rupees = (v) => "₹" + num(Math.round(v));
  const pct = (fraction) => num(fraction * 100, 0) + "%";

  const words = (s) =>
    String(s || "")
      .replace(/_/g, " ")
      .replace(/\b([a-z])/g, (m) => m.toUpperCase());

  const esc = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));

  /* ------------------------------------------------------------- fetching */

  async function getJSON(url, options) {
    const response = await fetch(url, options);
    let body = null;
    try {
      body = await response.json();
    } catch (err) {
      body = null;
    }
    if (!response.ok) {
      const detail = body && body.detail;
      throw new Error(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail) && detail.length
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : "Request failed with status " + response.status
      );
    }
    return body;
  }

  const postJSON = (url, payload) =>
    getJSON(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

  /* ----------------------------------------------------------------- form */

  function fillSelect(el, values, labels) {
    el.innerHTML = values
      .map(
        (v) =>
          '<option value="' +
          esc(v) +
          '">' +
          esc(labels && labels[v] ? labels[v] : words(v)) +
          "</option>"
      )
      .join("");
  }

  function buildForm(params) {
    fillSelect($("roof_type"), params.roof_types, {
      rcc_flat: "RCC flat slab",
      rcc_sloped: "RCC sloped slab",
      tiled: "Mangalore tile",
      gi_metal_sheet: "GI / metal sheet",
      asbestos_sheet: "Asbestos sheet",
    });
    $("roof_type").value = params.defaults.roof_type;

    fillSelect($("open_area_type"), params.open_area_types, {
      paved_concrete: "Concrete paving",
      paved_block: "Paver blocks",
      gravel: "Gravel",
      bare_soil: "Bare soil",
      lawn_garden: "Lawn or garden",
    });
    $("open_area_type").value = params.defaults.open_area_type;

    fillSelect($("house_type"), params.house_types, { row_house: "Row house" });
    fillSelect($("water_source"), params.water_sources, {
      cauvery: "Cauvery piped supply",
      borewell: "Borewell",
      tanker: "Tanker water",
      cauvery_and_borewell: "Cauvery + borewell",
    });
    fillSelect($("engine"), params.engine_modes, {
      auto: "Rules, with the model checked against them",
      rules: "Rules only",
      ml: "Trained classifier",
    });

    $("daily_use_lpcd").placeholder = params.defaults.lpcd + " (norm)";
    $("storage_design_days").placeholder = params.defaults.storage_design_days;
    $("meta-year").textContent = params.price_base_year;
  }

  function buildHobliOptions(hoblis) {
    const grouped = {};
    hoblis.forEach((h) => {
      (grouped[h.taluk] = grouped[h.taluk] || []).push(h);
    });
    const select = $("hobli");
    select.innerHTML =
      '<option value="">&mdash; select a hobli &mdash;</option>' +
      Object.keys(grouped)
        .sort()
        .map(
          (taluk) =>
            '<optgroup label="' +
            esc(taluk) +
            '">' +
            grouped[taluk]
              .map(
                (h) =>
                  '<option value="' + esc(h.hobli) + '">' + esc(h.hobli) + "</option>"
              )
              .join("") +
            "</optgroup>"
        )
        .join("");
  }

  /* ------------------------------------------------------------------ map */

  function initMap(hoblis, box) {
    if (typeof L === "undefined") {
      $("map-shell").classList.add("no-map");
      return;
    }
    const map = L.map("map", { zoomControl: true, scrollWheelZoom: false }).setView(
      [12.9716, 77.5946],
      10
    );
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    L.rectangle(
      [
        [box.min_lat, box.min_lon],
        [box.max_lat, box.max_lon],
      ],
      { color: "#176b7a", weight: 1, dashArray: "4 4", fill: false, interactive: false }
    ).addTo(map);

    hoblis.forEach((h) => {
      L.circleMarker([h.lat, h.lon], {
        radius: 4,
        color: "#0c3f4a",
        weight: 1,
        fillColor: "#8fd0d4",
        fillOpacity: 0.85,
      })
        .addTo(map)
        .bindTooltip(
          h.hobli + " · " + h.annual_rainfall_mm + " mm/yr",
          { direction: "top" }
        )
        .on("click", (event) => {
          setPin(h.lat, h.lon);
          L.DomEvent.stopPropagation(event);
        });
    });

    map.on("click", (event) => setPin(event.latlng.lat, event.latlng.lng));
    state.map = map;
  }

  function setPin(lat, lon) {
    state.lat = Number(lat.toFixed(5));
    state.lon = Number(lon.toFixed(5));
    $("hobli").value = "";
    $("pin-coords").textContent = state.lat.toFixed(4) + ", " + state.lon.toFixed(4);
    $("pin-hint").textContent = "pin set";

    if (state.map) {
      if (state.marker) {
        state.marker.setLatLng([lat, lon]);
      } else {
        state.marker = L.marker([lat, lon], { draggable: true }).addTo(state.map);
        state.marker.on("dragend", (event) => {
          const p = event.target.getLatLng();
          setPin(p.lat, p.lng);
        });
      }
    }
    loadSite();
  }

  /* ------------------------------------------------------------ site card */

  function locationQuery() {
    if (state.lat != null && state.lon != null) {
      return "lat=" + state.lat + "&lon=" + state.lon;
    }
    const hobli = $("hobli").value;
    return hobli ? "hobli=" + encodeURIComponent(hobli) : null;
  }

  async function loadSite() {
    const query = locationQuery();
    const card = $("site-card");
    if (!query) {
      card.hidden = true;
      return;
    }
    try {
      const site = await getJSON("/api/location?" + query);
      $("site-name").textContent = site.hobli + " · " + site.taluk;
      $("site-rain").textContent =
        site.annual_rainfall_mm + " mm/yr over " + site.annual_rainy_days + " days" +
        (site.rainfall_data_grade === "ksndmc_taluk_normal" ? " (taluk gauge)" : "");
      $("site-soil").textContent =
        words(site.soil_type) + " · " + site.infiltration_mm_per_hr + " mm/hr";
      $("site-gw").textContent = site.gw_depth_m + " m below ground";
      $("site-cat").textContent = words(site.gw_category);
      $("site-zone").textContent = site.bbmp_zone
        ? site.bbmp_zone + " · " + num(site.population_density_per_km2) + "/km²"
        : "outside BBMP";
      card.hidden = false;
    } catch (err) {
      $("site-name").textContent = "Location not covered";
      $("site-rain").textContent = err.message;
      ["site-soil", "site-gw", "site-cat", "site-zone"].forEach((id) => {
        $(id).textContent = "—";
      });
      card.hidden = false;
    }
  }

  /* ------------------------------------------------------------- payload */

  function numberOrNull(id) {
    const raw = $(id).value.trim();
    if (raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }

  function buildPayload() {
    const payload = {
      roof_area_m2: numberOrNull("roof_area_m2"),
      roof_type: $("roof_type").value,
      open_area_m2: numberOrNull("open_area_m2") || 0,
      open_area_type: $("open_area_type").value,
      household_size: numberOrNull("household_size"),
      house_type: $("house_type").value,
      water_source: $("water_source").value,
      has_borewell: $("has_borewell").checked,
      existing_storage_litres: numberOrNull("existing_storage_litres") || 0,
      slope_pct: numberOrNull("slope_pct") || 0,
      new_building: $("new_building").checked,
      engine: $("engine").value,
    };

    const optional = {
      daily_use_lpcd: numberOrNull("daily_use_lpcd"),
      plot_area_m2: numberOrNull("plot_area_m2"),
      storage_design_days: numberOrNull("storage_design_days"),
    };
    Object.keys(optional).forEach((key) => {
      if (optional[key] != null) payload[key] = optional[key];
    });

    if (state.lat != null && state.lon != null) {
      payload.lat = state.lat;
      payload.lon = state.lon;
    } else if ($("hobli").value) {
      payload.hobli = $("hobli").value;
    }
    return payload;
  }

  /* --------------------------------------------------------------- charts */

  function balanceChart(monthly) {
    const W = 760;
    const H = 280;
    const M = { top: 14, right: 14, bottom: 30, left: 54 };
    const plotW = W - M.left - M.right;
    const plotH = H - M.top - M.bottom;

    const peak = Math.max(
      1,
      ...monthly.map((m) =>
        Math.max(m.harvest_litres, m.demand_litres, m.closing_storage_litres)
      )
    );
    const step = plotW / monthly.length;
    const barW = Math.min(16, step / 3.2);
    const y = (value) => M.top + plotH - (value / peak) * plotH;

    const gridlines = [0, 0.25, 0.5, 0.75, 1]
      .map((f) => {
        const value = peak * f;
        const yy = y(value);
        return (
          '<line x1="' + M.left + '" y1="' + yy + '" x2="' + (W - M.right) + '" y2="' + yy +
          '" stroke="' + (f === 0 ? "#b9b1a0" : "#e3ddcf") + '" stroke-width="1"/>' +
          '<text x="' + (M.left - 8) + '" y="' + (yy + 3.5) +
          '" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="9" fill="#7b8d94">' +
          kl(value) + "</text>"
        );
      })
      .join("");

    const bars = monthly
      .map((m, i) => {
        const centre = M.left + step * i + step / 2;
        const hx = centre - barW - 1.5;
        const dx = centre + 1.5;
        const hy = y(m.harvest_litres);
        const dy = y(m.demand_litres);
        const base = y(0);
        return (
          '<rect x="' + hx + '" y="' + hy + '" width="' + barW + '" height="' +
          Math.max(0, base - hy) + '" fill="#176b7a"><title>' +
          esc(m.month_name) + " harvest " + num(m.harvest_litres) + " L</title></rect>" +
          '<rect x="' + dx + '" y="' + dy + '" width="' + barW + '" height="' +
          Math.max(0, base - dy) + '" fill="#d9bca9" stroke="#a9522f" stroke-width="0.75"><title>' +
          esc(m.month_name) + " demand " + num(m.demand_litres) + " L</title></rect>" +
          '<text x="' + centre + '" y="' + (H - 10) +
          '" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="9" fill="#465c66">' +
          esc(m.month_name) + "</text>"
        );
      })
      .join("");

    const points = monthly
      .map((m, i) => M.left + step * i + step / 2 + "," + y(m.closing_storage_litres))
      .join(" ");

    const dots = monthly
      .map(
        (m, i) =>
          '<circle cx="' + (M.left + step * i + step / 2) + '" cy="' +
          y(m.closing_storage_litres) + '" r="2.6" fill="#9a6c12"><title>' +
          esc(m.month_name) + " in store " + num(m.closing_storage_litres) + " L</title></circle>"
      )
      .join("");

    return (
      '<div class="chart-shell"><svg viewBox="0 0 ' + W + " " + H +
      '" role="img" aria-label="Monthly harvest, demand and stored water">' +
      gridlines +
      bars +
      '<polyline points="' + points +
      '" fill="none" stroke="#9a6c12" stroke-width="2" stroke-linejoin="round"/>' +
      dots +
      "</svg>" +
      '<div class="legend">' +
      '<span><i style="background:#176b7a"></i> Harvest</span>' +
      '<span><i style="background:#d9bca9;border:1px solid #a9522f"></i> Non-potable demand</span>' +
      '<span><i class="line"></i> Water left in store at month end</span>' +
      '<span style="margin-left:auto">kilolitres</span>' +
      "</div></div>"
    );
  }

  /* -------------------------------------------------------------- report */

  function tile(key, value, sub, tone) {
    return (
      '<div class="tile' + (tone ? " " + tone : "") + '"><span class="k">' + esc(key) +
      '</span><span class="v">' + value + '</span><span class="s">' + esc(sub) + "</span></div>"
    );
  }

  function sectionTitle(idx, heading, aside) {
    return (
      '<div class="section-title"><span class="idx">' + idx + "</span><h3>" + esc(heading) +
      "</h3>" + (aside ? '<span class="aside">' + esc(aside) + "</span>" : "") + "</div>"
    );
  }

  function specList(heading, why, rows) {
    return (
      '<div class="spec"><h4>' + esc(heading) + "</h4>" +
      (why ? '<p class="why">' + esc(why) + "</p>" : "") +
      "<dl>" +
      rows
        .map((r) => "<dt>" + esc(r[0]) + "</dt><dd>" + r[1] + "</dd>")
        .join("") +
      "</dl></div>"
    );
  }

  function renderHeadline(d) {
    const rec = d.recommendation;
    const balance = d.water_balance;
    const savings = d.savings;
    const harvest = d.harvest_potential;

    const tiles = [
      tile(
        "Annual harvest",
        kl(harvest.annual_roof_harvest_litres) + " <small>kL</small>",
        "from " + harvest.annual_rainfall_mm + " mm of rain"
      ),
      tile(
        "Demand met",
        num(balance.reliability_pct, 0) + " <small>%</small>",
        balance.months_fully_met + " of 12 months met in full",
        balance.reliability_pct >= 50 ? "good" : null
      ),
      tile(
        "Capital cost",
        d.cost.total_capital_cost_inr ? rupees(d.cost.total_capital_cost_inr) : "—",
        d.cost.total_capital_cost_inr
          ? rupees(d.cost.indicative_range_inr[0]) + " to " + rupees(d.cost.indicative_range_inr[1])
          : "nothing to build"
      ),
    ];

    if (savings) {
      tiles.push(
        tile(
          "Net saving",
          rupees(savings.annual_net_saving_inr) + " <small>/yr</small>",
          savings.pays_back_within_life
            ? "pays back in " + savings.payback_years + " years"
            : "does not pay back in " + savings.structure_life_years + " years",
          savings.annual_net_saving_inr > 0 ? "good" : "warn"
        ),
        tile(
          "Returned to the ground",
          kl(balance.annual_recharge_litres) + " <small>kL</small>",
          "aquifer is " + words(d.site.gw_category).toLowerCase()
        )
      );
    }

    return (
      '<section class="block"><div class="verdict">' +
      '<span class="pill' + (rec.label === "not_recommended" ? " warn" : "") + '">Recommended</span>' +
      '<h3 class="structure" style="margin-top:10px">' + esc(rec.title) + "</h3>" +
      '<p class="confidence">confidence ' + pct(rec.confidence) +
      " · decided by " + esc(d.meta.engine_used === "ml" ? "trained classifier" : "rule engine") +
      " · scores storage " + rec.scores.storage + " / pit " + rec.scores.recharge_pit +
      " / borewell " + rec.scores.borewell_recharge + "</p>" +
      '<ul class="summary">' +
      d.summary.map((s) => "<li>" + esc(s) + "</li>").join("") +
      "</ul></div>" +
      '<div class="tiles">' + tiles.join("") + "</div></section>"
    );
  }

  function renderWhy(d) {
    const rec = d.recommendation;
    const rejected = rec.rejected_options.length
      ? '<div class="rejected">' +
        sectionTitle("", "Options not taken", "") +
        '<ul class="reasons">' +
        rec.rejected_options
          .map(
            (r) =>
              '<li><span class="opt">' + esc(r.title) + "</span> &mdash; " + esc(r.why) + "</li>"
          )
          .join("") +
        "</ul></div>"
      : "";

    return (
      '<section class="block">' +
      sectionTitle("01", "Why this structure", rec.reasons.length + " factors") +
      '<ul class="reasons">' +
      rec.reasons.map((r) => "<li>" + esc(r) + "</li>").join("") +
      "</ul>" +
      rejected +
      "</section>"
    );
  }

  function renderSpecs(d) {
    const s = d.sizing;
    const cards = [];

    if (s.storage) {
      cards.push(
        specList(
          "Storage: " + num(s.storage.recommended_litres) + " litres",
          s.storage.material + " — " + s.storage.basis,
          [
            ["Recommended capacity", num(s.storage.recommended_litres) + " L"],
            ["Volume", s.storage.recommended_m3 + " m³"],
            [
              "To be built",
              s.storage.additional_litres
                ? num(s.storage.additional_litres) + " L"
                : "nothing — the existing sump suffices",
            ],
            ["Dry-spell cover", s.storage.design_days + " days"],
            ["Demand over that spell", num(s.storage.demand_based_litres) + " L"],
            ["Wettest month can deliver", num(s.storage.peak_month_harvest_litres) + " L"],
            [
              "Statutory minimum",
              s.storage.byelaw_minimum_litres
                ? num(s.storage.byelaw_minimum_litres) + " L"
                : "not applicable",
            ],
          ]
        )
      );
    }

    if (s.recharge_pit) {
      const p = s.recharge_pit;
      cards.push(
        specList(
          p.pits + " recharge pit" + (p.pits > 1 ? "s" : ""),
          p.reason,
          [
            ["Each pit", p.diameter_m + " m dia × " + p.depth_m + " m deep"],
            ["Total excavation", p.total_gross_volume_m3 + " m³"],
            ["Holds in the media voids", num(p.effective_storage_litres) + " L"],
            ["Design-storm runoff to take", num(p.design_storm_runoff_litres) + " L"],
            ["Soaks away during the storm", num(p.infiltration_during_storm_litres) + " L"],
            ["Empties in", p.drain_down_hours + " hours"],
            ["Recharge per year", kl(p.annual_recharge_litres) + " kL"],
          ]
        )
      );
    }

    if (s.borewell_recharge) {
      const b = s.borewell_recharge;
      cards.push(
        specList(
          b.units + " borewell recharge unit" + (b.units > 1 ? "s" : ""),
          b.reason,
          [
            ["Filter chamber", b.chamber_diameter_m + " m dia × " + b.chamber_depth_m + " m deep"],
            ["Settling time", b.retention_minutes + " minutes of design storm"],
            ["Volume needed", b.required_volume_m3 + " m³"],
            ["Volume provided", b.provided_volume_m3 + " m³"],
            ["Recharge per year", kl(b.annual_recharge_litres) + " kL"],
          ]
        )
      );
    }

    if (!cards.length) return "";

    return (
      '<section class="block">' +
      sectionTitle("02", "What to build", "sizes from the design storm and the dry spell") +
      '<div class="two-col">' + cards.join("") + "</div></section>"
    );
  }

  function renderBalance(d) {
    const b = d.water_balance;
    const rain = d.site.monthly_rainfall;

    const rows = b.monthly
      .map((m, i) => {
        const r = rain[i] || {};
        return (
          "<tr><td>" + esc(m.month_name) + "</td>" +
          '<td class="n">' + num(r.rainfall_mm || 0, 1) + "</td>" +
          '<td class="n">' + num(m.harvest_litres) + "</td>" +
          '<td class="n">' + num(m.demand_litres) + "</td>" +
          '<td class="n">' + num(m.supplied_litres) + "</td>" +
          '<td class="n">' + num(m.deficit_litres) + "</td>" +
          '<td class="n">' + num(m.overflow_litres) + "</td>" +
          '<td class="n">' + num(m.closing_storage_litres) + "</td></tr>"
        );
      })
      .join("");

    return (
      '<section class="block">' +
      sectionTitle("03", "Supply against demand", "monthly balance, settled year") +
      balanceChart(b.monthly) +
      '<div class="table-wrap" style="margin-top:18px"><table class="data">' +
      "<caption>All volumes in litres</caption>" +
      "<thead><tr><th>Month</th><th class='n'>Rain mm</th><th class='n'>Harvest</th>" +
      "<th class='n'>Demand</th><th class='n'>Supplied</th><th class='n'>Shortfall</th>" +
      "<th class='n'>Overflow</th><th class='n'>In store</th></tr></thead>" +
      "<tbody>" + rows + "</tbody>" +
      "<tfoot><tr><td>Year</td><td class='n'>" + num(d.harvest_potential.annual_rainfall_mm, 1) +
      "</td><td class='n'>" + num(b.annual_harvest_litres) + "</td><td class='n'>" +
      num(b.annual_non_potable_demand_litres) + "</td><td class='n'>" +
      num(b.annual_supplied_litres) + "</td><td class='n'>" + num(b.annual_deficit_litres) +
      "</td><td class='n'>" + num(b.annual_overflow_litres) + "</td><td class='n'>—</td></tr></tfoot>" +
      "</table></div>" +
      '<div class="callout info">Rainwater covers <strong>' + num(b.reliability_pct, 1) +
      "%</strong> of the household's non-potable demand over the year. The rest still has to come " +
      "from the existing supply, and the shortfall falls in the dry months from January to May.</div>" +
      "</section>"
    );
  }

  /**
   * What the recommended system does in a weak monsoon. Shown as a comparison
   * against the normal year rather than on its own, because the number only
   * means something next to the headline figure it qualifies.
   */
  function renderDryYear(d) {
    const dy = d.dry_year_outlook;
    if (!dy) return "";

    const rows = dy.monthly
      .map(
        (m) =>
          "<tr><td>" + esc(m.month_name) + "</td>" +
          '<td class="n">' + num(m.rainfall_mm, 1) + "</td>" +
          '<td class="n">' + pct(m.ratio_to_normal) + "</td>" +
          '<td class="n">' + num(m.harvest_litres) + "</td>" +
          '<td class="n">' + num(m.supplied_litres) + "</td>" +
          '<td class="n">' + num(m.deficit_litres) + "</td></tr>"
      )
      .join("");

    const drop = dy.reliability_drop_pct_points;
    const tone = drop >= 20 ? "bad" : drop >= 10 ? "info" : "ok";
    const p = dy.annual_percentiles_mm;

    return (
      '<section class="block">' +
      sectionTitle("04", "If the monsoon is weak", "same system, drier year") +
      '<div class="callout ' + tone + '"><strong>In a dry year this system covers ' +
      num(dy.reliability_pct, 0) + "% of non-potable demand instead of " +
      num(dy.reliability_pct_normal_year, 0) + "%</strong> — a fall of " +
      num(drop, 0) + " percentage points, with " + dy.months_fully_met +
      " of 12 months met in full rather than " + dy.months_fully_met_normal_year +
      ". " + esc(dy.note) + "</div>" +
      '<div class="table-wrap" style="margin-top:14px"><table class="data">' +
      "<caption>Volumes in litres. The tank is not resized — this is the same build, a drier year." +
      "</caption>" +
      "<thead><tr><th>Month</th><th class='n'>Rain mm</th><th class='n'>Of normal</th>" +
      "<th class='n'>Harvest</th><th class='n'>Supplied</th><th class='n'>Shortfall</th>" +
      "</tr></thead><tbody>" + rows + "</tbody>" +
      "<tfoot><tr><td>Year</td><td class='n'>" + num(dy.annual_rainfall_mm, 1) +
      "</td><td class='n'>" + pct(dy.annual_fraction_of_normal) + "</td><td class='n'>" +
      num(dy.annual_roof_harvest_litres) + "</td><td class='n'>" +
      num(dy.annual_supplied_litres) + "</td><td class='n'>" +
      num(dy.annual_deficit_litres) + "</td></tr></tfoot></table></div>" +
      '<p class="fineprint" style="margin-top:8px">' + esc(dy.scenario) +
      " Across the full record annual totals run " + num(p.p10, 0) + " mm at the 10th " +
      "percentile, " + num(p.p50, 0) + " mm at the median and " + num(p.p90, 0) +
      " mm at the 90th.</p></section>"
    );
  }

  function renderCost(d) {
    if (!d.cost.lines.length) return "";
    const c = d.cost;
    const s = d.savings;

    const lines = c.lines
      .map(
        (l) =>
          "<tr><td>" + esc(l.description) + "</td>" +
          '<td class="n">' + num(l.quantity, l.quantity < 100 ? 1 : 0) + " " + esc(l.unit) + "</td>" +
          '<td class="n">' + rupees(l.rate_inr) + "</td>" +
          '<td class="n">' + rupees(l.amount_inr) + "</td></tr>"
      )
      .join("");

    const savingsBlock = s
      ? specList("Water saved and what it is worth", s.rate_basis, [
          ["Water used from the tank", kl(s.annual_water_saved_litres) + " kL/yr"],
          ["Water returned to the aquifer", kl(s.annual_recharge_litres) + " kL/yr"],
          ["Value of water displaced", rupees(s.annual_gross_saving_inr) + "/yr"],
          ["Upkeep", "− " + rupees(c.annual_maintenance_inr) + "/yr"],
          ["Net saving", "<strong>" + rupees(s.annual_net_saving_inr) + "/yr</strong>"],
          ["Tanker loads avoided", num(s.tanker_loads_avoided_per_year, 1) + "/yr"],
          ["Payback", s.payback_years ? s.payback_years + " years" : "not within its life"],
          [
            "Over " + s.structure_life_years + " years",
            rupees(s.lifetime_net_benefit_inr),
          ],
        ])
      : "";

    const noteClass = !s
      ? "info"
      : s.pays_back_within_life
      ? "ok"
      : s.annual_net_saving_inr > 0
      ? ""
      : "bad";

    return (
      '<section class="block">' +
      sectionTitle("05", "Cost and saving", "indicative " + d.meta.price_base_year + " rates") +
      '<div class="two-col">' +
      '<div><div class="table-wrap"><table class="data">' +
      "<caption>Capital cost</caption>" +
      "<thead><tr><th>Item</th><th class='n'>Quantity</th><th class='n'>Rate</th><th class='n'>Amount</th></tr></thead>" +
      "<tbody>" + lines +
      '<tr class="sub"><td colspan="3">Plumbing and labour</td><td class="n">' +
      rupees(c.plumbing_and_labour_inr) + "</td></tr>" +
      '<tr class="sub"><td colspan="3">Contingency</td><td class="n">' +
      rupees(c.contingency_inr) + "</td></tr></tbody>" +
      '<tfoot><tr><td colspan="3">Total</td><td class="n">' +
      rupees(c.total_capital_cost_inr) + "</td></tr></tfoot>" +
      "</table></div>" +
      '<p class="fineprint" style="margin-top:8px">Likely range ' +
      rupees(c.indicative_range_inr[0]) + " to " + rupees(c.indicative_range_inr[1]) +
      ", depending on site access, material prices and contractor.</p></div>" +
      "<div>" + savingsBlock +
      (s ? '<div class="callout ' + noteClass + '">' + esc(s.note) + "</div>" : "") +
      "</div></div></section>"
    );
  }

  function renderByelaw(d) {
    const b = d.byelaw_check;
    const tone =
      b.applicable === null ? "info" : b.compliant === false ? "bad" : b.applicable ? "ok" : "info";
    const verdict =
      b.applicable === null
        ? "Applicability unknown"
        : !b.applicable
        ? "Voluntary on this plot"
        : b.compliant
        ? "The recommended system meets the minimum"
        : "The recommended system falls short of the minimum";

    return (
      '<section class="block">' +
      sectionTitle("06", "Byelaw check", b.reference) +
      '<div class="callout ' + tone + '"><strong>' + esc(verdict) + ".</strong> " +
      esc(b.reason) + "</div>" +
      '<div class="table-wrap" style="margin-top:14px"><table class="data">' +
      "<thead><tr><th>Provision</th><th class='n'>Required</th><th class='n'>Provided</th></tr></thead><tbody>" +
      "<tr><td>Storage per m² of roof</td><td class='n'>" + b.storage_minimum_litres_per_m2_roof +
      " L</td><td class='n'>" + num(b.storage_provided_litres_per_m2_roof, 1) + " L</td></tr>" +
      "<tr><td>Recharge per m² of roof</td><td class='n'>" + b.recharge_minimum_litres_per_m2_roof +
      " L</td><td class='n'>" + num(b.recharge_provided_litres_per_m2_roof, 1) + " L</td></tr>" +
      "</tbody></table></div>" +
      '<p class="fineprint" style="margin-top:8px">Either provision satisfies the rule. Confirm the ' +
      "figures against the current notification before using them in a plan sanction.</p></section>"
    );
  }

  function renderModel(d) {
    const ml = d.ml;
    const rows = [["Engine requested", esc(words(ml.mode))], ["Decision taken from", esc(ml.used === "ml" ? "trained classifier" : "rule engine")]];

    if (ml.prediction) {
      rows.push(
        ["Classifier", esc(ml.prediction.model)],
        ["Classifier label", esc(words(ml.prediction.label))],
        ["Classifier confidence", pct(ml.prediction.probability)],
        ["Agrees with the rules", ml.agrees_with_rules ? "yes" : "no"]
      );
    }
    if (state.model && state.model.metrics) {
      const m = state.model.metrics;
      if (m.test_accuracy) rows.push(["Holdout accuracy", pct(m.test_accuracy)]);
      if (m.test_f1_macro) rows.push(["Macro F1 (all labels equally)", pct(m.test_f1_macro)]);
      if (m.n_samples) rows.push(["Trained on", num(m.n_samples) + " labelled sites"]);
      if (typeof m.calibrated === "boolean") {
        rows.push([
          "Confidence calibrated",
          m.calibrated
            ? "yes, isotonic (error " + num(m.expected_calibration_error * 100, 1) + "%)"
            : "no — read the confidence as a ranking only",
        ]);
      }
    }

    const importances =
      state.model && state.model.feature_importances
        ? state.model.feature_importances.slice(0, 6)
        : [];

    const bars = importances.length
      ? '<div><div class="section-title" style="border:0;margin-bottom:6px"><h4 style="font-size:.92rem">What drives the choice</h4></div>' +
        '<table class="data"><tbody>' +
        importances
          .map(
            (f) =>
              "<tr><td>" + esc(f.label) + "</td><td style='width:45%'>" +
              '<span style="display:inline-block;height:8px;background:#176b7a;width:' +
              (f.importance * 100).toFixed(1) + '%"></span></td>' +
              '<td class="n">' + num(f.importance * 100, 1) + "%</td></tr>"
          )
          .join("") +
        "</tbody></table>" +
        '<p class="fineprint" style="margin-top:6px">Random-forest feature importances across the ' +
        "whole training set, not for this one site.</p></div>"
      : "";

    return (
      '<section class="block">' +
      sectionTitle("07", "Model and rules", "the two are cross-checked") +
      '<div class="two-col">' +
      specList("This request", ml.note || "", rows) +
      bars +
      "</div></section>"
    );
  }

  function renderNotes(d) {
    return (
      '<section class="block">' +
      sectionTitle("08", "Upkeep and limits", "read before building") +
      '<div class="two-col">' +
      '<div><h4 style="font-size:.95rem;margin-bottom:8px">Keeping it working</h4>' +
      '<ul class="note-list">' +
      d.maintenance.map((m) => "<li>" + esc(m) + "</li>").join("") +
      "</ul></div>" +
      '<div><h4 style="font-size:.95rem;margin-bottom:8px">What this does not tell you</h4>' +
      '<ul class="note-list">' +
      d.caveats.map((c) => "<li>" + esc(c) + "</li>").join("") +
      "</ul></div></div></section>"
    );
  }

  /**
   * One sentence on where this site's rainfall actually came from. Kept separate
   * from the soil and groundwater wording because the two differ in kind: the
   * rainfall is a measured gauge normal, the rest are estimates.
   */
  function rainfallProvenance(site) {
    const grade = site.rainfall_data_grade || "";
    if (grade === "ksndmc_hobli_normal") {
      return (
        "Rainfall is the KSNDMC rain-gauge normal for " + esc(site.hobli) +
        ", including its measured split across the three monsoon seasons; the shape within " +
        "each season follows the IMD Bengaluru City monthly normals."
      );
    }
    if (grade === "ksndmc_taluk_normal") {
      return (
        "Rainfall is the KSNDMC normal for " + esc(site.taluk) + " taluk - " + esc(site.hobli) +
        " has no gauge of its own - so it is measured but averaged over a wider area."
      );
    }
    return "Rainfall is a reference estimate, not a gauge reading.";
  }

  function renderReport(d) {
    state.lastReport = d;
    const site = d.site;
    const now = new Date().toLocaleDateString("en-IN", {
      day: "numeric",
      month: "long",
      year: "numeric",
    });

    const head =
      '<div class="report-head">' +
      "<div><span class='label'>Assessment sheet</span>" +
      "<h2>" + esc(site.hobli) + "</h2>" +
      '<span class="where">' + esc(site.taluk) + " taluk · " + esc(site.district) +
      (site.bbmp_zone ? " · BBMP " + esc(site.bbmp_zone) + " zone" : "") +
      " · " + esc(now) + "</span></div>" +
      '<div class="spacer no-print">' +
      '<button type="button" class="ghost" id="print-btn">Print this sheet</button></div>' +
      "</div>";

    $("empty-state").hidden = true;
    $("report").innerHTML =
      head +
      renderHeadline(d) +
      renderWhy(d) +
      renderSpecs(d) +
      renderBalance(d) +
      renderDryYear(d) +
      renderCost(d) +
      renderByelaw(d) +
      renderModel(d) +
      renderNotes(d) +
      '<div class="sheet-foot">' +
      rainfallProvenance(site) +
      " Soil from NBSS&amp;LUP/SoilGrids and groundwater from CGWB/India-WRIS are hobli-level " +
      "reference estimates, not site measurements. Boundaries and basemap from OpenStreetMap. " +
      "Generated by the recommendation engine v" + esc(d.meta.app_version) + ".</div>";

    const printBtn = $("print-btn");
    if (printBtn) printBtn.addEventListener("click", () => window.print());

    $("sheet").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(message) {
    $("empty-state").hidden = true;
    $("report").innerHTML =
      '<div class="error"><strong>Could not produce a recommendation.</strong><br>' +
      esc(message) + "</div>" + ($("report").dataset.keep || "");
  }

  /* --------------------------------------------------------------- events */

  async function onSubmit(event) {
    event.preventDefault();
    const payload = buildPayload();

    if (!payload.lat && !payload.hobli) {
      showError("Set a pin on the map or choose a hobli first.");
      return;
    }
    if (!payload.roof_area_m2 || payload.roof_area_m2 <= 0) {
      showError("Enter the roof area in square metres.");
      return;
    }
    if (!payload.household_size || payload.household_size < 1) {
      showError("Enter how many people live in the house.");
      return;
    }

    const button = $("submit-btn");
    button.disabled = true;
    button.textContent = "Working…";
    try {
      renderReport(await postJSON("/api/recommend", payload));
    } catch (err) {
      showError(err.message);
    } finally {
      button.disabled = false;
      button.textContent = "Get recommendation";
    }
  }

  function onReset() {
    $("site-form").reset();
    state.lat = null;
    state.lon = null;
    if (state.marker && state.map) {
      state.map.removeLayer(state.marker);
      state.marker = null;
    }
    $("pin-coords").textContent = "no pin set";
    $("pin-hint").textContent = "or choose a hobli";
    $("site-card").hidden = true;
    $("report").innerHTML = "";
    $("empty-state").hidden = false;
  }

  /* ----------------------------------------------------------------- init */

  /** Keep the sticky rail's height in step with the real masthead height. */
  function syncHeaderHeight() {
    const masthead = document.querySelector(".masthead");
    if (!masthead) return;
    const height = Math.ceil(masthead.getBoundingClientRect().height);
    document.documentElement.style.setProperty("--head", height + "px");
  }

  async function init() {
    syncHeaderHeight();
    window.addEventListener("resize", syncHeaderHeight);
    $("site-form").addEventListener("submit", onSubmit);
    $("reset-btn").addEventListener("click", onReset);
    $("hobli").addEventListener("change", () => {
      if ($("hobli").value) {
        state.lat = null;
        state.lon = null;
        $("pin-coords").textContent = "using hobli centroid";
        $("pin-hint").textContent = $("hobli").value;
        if (state.marker && state.map) {
          state.map.removeLayer(state.marker);
          state.marker = null;
        }
      }
      loadSite();
    });

    try {
      const [params, hoblis] = await Promise.all([
        getJSON("/api/parameters"),
        getJSON("/api/hoblis"),
      ]);
      state.params = params;
      state.hoblis = hoblis.hoblis;
      buildForm(params);
      buildHobliOptions(state.hoblis);
      $("meta-hoblis").textContent = hoblis.count;
      initMap(state.hoblis, params.bounding_box);
    } catch (err) {
      showError(
        "The service did not respond: " + err.message +
          ". Start it with: uvicorn app.main:app --reload"
      );
      return;
    }

    try {
      const model = await getJSON("/api/model");
      state.model = model;
      $("meta-model").textContent = model.available
        ? model.model + " · " + pct(model.metrics.test_accuracy || 0)
        : "rules only";
    } catch (err) {
      $("meta-model").textContent = "rules only";
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
