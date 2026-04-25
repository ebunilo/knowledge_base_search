/* global fetch, document, localStorage, window */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const state = {
    user: { user_id: "anonymous" },
    profile: "",
  };

  function getTheme() {
    return localStorage.getItem("kb-theme") || "dark";
  }
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("kb-theme", t);
    const btn = $("btnTheme");
    if (btn) {
      btn.setAttribute("aria-pressed", t === "light" ? "true" : "false");
      btn.textContent = t === "light" ? "◑" : "◐";
    }
  }

  function collectUserPayload() {
    const sel = $("userSelect");
    const v = (sel && sel.value) || "anonymous";
    return { user_id: v === "anonymous" ? "anonymous" : v };
  }

  function getCommonReq() {
    return {
      user: collectUserPayload(),
      top_k: parseInt($("topK").value, 10) || 8,
      rerank: $("optRerank").checked,
      rewrite: $("rewrite").value,
      multi_query_k: 2,
      stepback: $("optStepback").checked,
    };
  }

  function updateProfileLabel() {
    const sel = $("userSelect");
    if (!sel) return;
    const o = sel.options[sel.selectedIndex];
    state.profile = o ? o.text : "";
    const el = $("profileLabel");
    if (el) el.textContent = state.profile || "—";
  }

  function loadConfig() {
    return fetch("/api/config")
      .then((r) => r.json())
      .then((data) => {
        const sel = $("userSelect");
        if (!sel) return;
        sel.innerHTML = "";
        (data.users || []).forEach((u) => {
          const opt = document.createElement("option");
          opt.value = u.id;
          opt.textContent = u.label;
          opt.dataset.department = u.department || "";
          opt.dataset.role = u.role || "";
          sel.appendChild(opt);
        });
        const saved = localStorage.getItem("kb-user-id");
        if (saved && Array.from(sel.options).some((o) => o.value === saved)) {
          sel.value = saved;
        }
        sel.addEventListener("change", () => {
          localStorage.setItem("kb-user-id", sel.value);
          updateProfileLabel();
        });
        updateProfileLabel();
        const pl = $("profileLabel");
        if (pl) pl.title = "Profile: " + (data.app_profile || "");
      })
      .catch(() => {
        const sel = $("userSelect");
        if (sel) {
          sel.innerHTML = "<option value=\"anonymous\">Anonymous</option>";
        }
      });
  }

  function healthTick() {
    return fetch("/api/health")
      .then((r) => r.json())
      .then((d) => {
        const p = $("healthPill");
        if (!p) return;
        if (d.ok) {
          p.setAttribute("data-state", "ok");
          p.title = "Data plane: OK (postgres, qdrant, bm25)";
        } else {
          p.setAttribute("data-state", "err");
          p.title =
            "Data plane: check services — " +
            JSON.stringify(d.services || {}, null, 0);
        }
      })
      .catch(() => {
        const p = $("healthPill");
        if (p) p.setAttribute("data-state", "err");
      });
  }

  /* ——— Tabs ——— */
  function setTab(name) {
    const askP = $("panelAsk");
    const seP = $("panelSearch");
    const a = $("tabAsk");
    const s = $("tabSearch");
    if (name === "ask") {
      askP.classList.add("active");
      askP.hidden = false;
      seP.classList.remove("active");
      seP.hidden = true;
      a.classList.add("active");
      a.setAttribute("aria-selected", "true");
      s.classList.remove("active");
      s.setAttribute("aria-selected", "false");
    } else {
      seP.classList.add("active");
      seP.hidden = false;
      askP.classList.remove("active");
      askP.hidden = true;
      s.classList.add("active");
      s.setAttribute("aria-selected", "true");
      a.classList.remove("active");
      a.setAttribute("aria-selected", "false");
    }
  }

  /* ——— Search ——— */
  function runSearch(e) {
    e.preventDefault();
    const q = ($("searchInput").value || "").trim();
    const ul = $("searchResults");
    const meta = $("searchMeta");
    if (!q) return;
    ul.innerHTML = "";
    meta.textContent = "Searching…";
    const body = {
      query: q,
      user: collectUserPayload(),
      top_k: parseInt($("topK").value, 10) || 8,
      rerank: $("optRerank").checked,
      rewrite: $("rewrite").value,
      multi_query_k: 2,
    };
    fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => {
        if (r.ok) return r.json();
        return r.text().then((t) => {
          try {
            return Promise.reject(t ? JSON.parse(t) : { message: r.statusText });
          } catch {
            return Promise.reject({ message: t || r.statusText });
          }
        });
      })
      .then((res) => {
        meta.textContent = `${res.hits && res.hits.length ? res.hits.length : 0} result(s) · ${
          res.total_ms
        } ms total`;
        const hits = res.hits || [];
        if (!hits.length) {
          ul.innerHTML = "<li><p class=\"hint\">No hits. Try a different phrasing or check ingestion.</p></li>";
          return;
        }
        hits.forEach((h, i) => {
          const li = document.createElement("li");
          const d = h.content || "";
          const sn = d.length > 320 ? d.slice(0, 320) + "…" : d;
          li.innerHTML = `
            <div class="card">
              <h3>${(h.title || "(no title)").replace(/</g, "&lt;")}</h3>
              <p class="sub">${(h.source_id || "").replace(/</g, "&lt;")} · ${(h.section_path || "/").replace(
                /</g,
                "&lt;"
              )}</p>
              <p class="snip">${sn.replace(/</g, "&lt;")}</p>
              <p class="score-bar">#${i + 1} · score ${Number(h.score).toFixed(4)}${
            h.rerank_rank ? " · rerank " + h.rerank_rank : ""
          }</p>
            </div>
          `;
          ul.appendChild(li);
        });
      })
      .catch((err) => {
        meta.textContent = "";
        ul.innerHTML = `<li><div class="err-box">Search failed: ${
          err.detail || err.error || err.message || String(err)
        }</div></li>`;
      });
  }

  /* ——— Chat ——— */
  function elBubble(role, text, metaHtml) {
    const wrap = document.createElement("div");
    wrap.className = "bubble " + role;
    const lab = document.createElement("p");
    lab.className = "label";
    lab.textContent = role === "user" ? "You" : "Answer";
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text;
    wrap.appendChild(lab);
    wrap.appendChild(body);
    if (metaHtml) {
      const m = document.createElement("div");
      m.className = "meta";
      m.innerHTML = metaHtml;
      wrap.appendChild(m);
    }
    return wrap;
  }

  function formatDoneResult(result) {
    if (!result) return "";
    const c = result.citations || [];
    const parts = [];
    if (c.length) {
      const ul = c
        .map(
          (x) =>
            `<li>[${x.marker}] ${
              (x.title || "").replace(/</g, "&lt;")
            } <span class="sub">(${(x.source_id || "").replace(/</g, "&lt;")})</span></li>`
        )
        .join("");
      parts.push(`<strong>Citations</strong><ul class="citation-list">${ul}</ul>`);
    }
    if (result.retrieval && result.retrieval.resolved_query) {
      parts.push(
        "Resolved: " + (result.retrieval.resolved_query || "").replace(/</g, "&lt;")
      );
    }
    parts.push(
      `confidence ${(result.confidence != null ? result.confidence : 0).toFixed(2)}` +
        " · " +
        `gen ${
          result.generation_ms != null ? result.generation_ms : 0
        } ms` +
        (result.faithfulness
          ? " · NLI " + (result.faithfulness.supported_ratio * 100).toFixed(0) + "% supported"
          : "")
    );
    return parts.join("<br/>");
  }

  function formatDoneMetaNode(result) {
    const n = document.createElement("div");
    n.className = "meta-inner";
    n.innerHTML = formatDoneResult(result);
    return n;
  }

  function runAskStream(userText) {
    const log = $("chatLog");
    const u = elBubble("user", userText, null);
    log.appendChild(u);
    const a = elBubble("assistant", "", null);
    const body = a.querySelector(".body");
    a.querySelector(".label").textContent = "Answer (streaming)…";
    log.appendChild(a);
    log.scrollTop = log.scrollHeight;

    const pay = {
      ...getCommonReq(),
      query: userText,
      check_faithfulness: $("optFaith").checked,
      session_id: ($("sessionInput").value || "").trim() || null,
    };
    if (!pay.check_faithfulness) pay.check_faithfulness = false;
    if (!pay.session_id) delete pay.session_id;

    const handleEvent = (j) => {
      if (j == null) return;
      if (j.kind === "error" && j.message) {
        body.textContent = "Error: " + j.message;
        return;
      }
      if (j.kind === "token" && j.text) {
        if (!a.dataset.acc) a.dataset.acc = "";
        a.dataset.acc += j.text;
        body.textContent = a.dataset.acc;
        a.querySelector(".label").textContent = "Answer";
        log.scrollTop = log.scrollHeight;
        return;
      }
      if (j.kind === "refused" && j.text) {
        body.textContent = j.text;
        a.querySelector(".label").textContent = "Refused";
        return;
      }
      if (j.kind === "refused" && j.result) {
        body.textContent = j.result.answer || "";
        a.querySelector(".label").textContent = "Refused";
        const div = document.createElement("div");
        div.className = "meta";
        div.textContent = j.result.refusal_reason || "";
        a.appendChild(div);
        return;
      }
      if (j.kind === "done" && j.result) {
        const res = j.result;
        a.querySelector(".label").textContent = "Answer";
        body.textContent = res.answer || a.dataset.acc || "";
        if (res.session_id) {
          $("sessionInput").value = res.session_id;
        }
        const m = a.querySelector(".meta");
        if (m) m.remove();
        const div = document.createElement("div");
        div.className = "meta";
        div.appendChild(formatDoneMetaNode(res));
        a.appendChild(div);
      }
    };

    let buf = "";
    function consumeSseBuffer() {
      buf = buf.replace(/\r\n/g, "\n");
      const events = buf.split("\n\n");
      buf = events.pop() || "";
      events.forEach((block) => {
        block.split("\n").forEach((line) => {
          if (!line.startsWith("data: ")) return;
          const data = line.slice(6).trim();
          if (data === "[DONE]") return;
          let j;
          try {
            j = JSON.parse(data);
          } catch {
            return;
          }
          handleEvent(j);
        });
      });
    }
    fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pay),
    })
      .then(function (r) {
        if (!r.ok) {
          return r.text().then((t) => {
            throw new Error(t || r.statusText);
          });
        }
        if (!r.body) throw new Error("no body");
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        return (function readLoop() {
          return reader.read().then(function (pr) {
            if (pr.done) {
              consumeSseBuffer();
              return;
            }
            buf += dec.decode(pr.value, { stream: true });
            consumeSseBuffer();
            return readLoop();
          });
        })();
      })
      .catch(function (e) {
        a.querySelector(".label").textContent = "Error";
        body.textContent = String(e && e.message ? e.message : e);
      });
  }

  function runAskSync(userText) {
    const log = $("chatLog");
    log.appendChild(elBubble("user", userText, null));
    const a = elBubble("assistant", "…", null);
    log.appendChild(a);
    const body = a.querySelector(".body");
    const pay = {
      ...getCommonReq(),
      query: userText,
      check_faithfulness: $("optFaith").checked,
      session_id: ($("sessionInput").value || "").trim() || null,
    };
    if (!pay.session_id) delete pay.session_id;
    if (!pay.check_faithfulness) pay.check_faithfulness = false;

    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pay),
    })
      .then((r) => {
        if (r.ok) return r.json();
        return r.text().then((t) => {
          try {
            return Promise.reject(t ? JSON.parse(t) : { message: r.statusText });
          } catch {
            return Promise.reject({ message: t || r.statusText });
          }
        });
      })
      .then((res) => {
        body.textContent = res.answer || (res.refused ? res.refusal_reason : "");
        if (res.session_id) {
          $("sessionInput").value = res.session_id;
        }
        const m = document.createElement("div");
        m.className = "meta";
        m.innerHTML = formatDoneResult(res);
        a.appendChild(m);
        log.scrollTop = log.scrollHeight;
      })
      .catch((err) => {
        body.textContent = "Error: " + (err.detail || err.error || err.message);
      });
  }

  function onAskSubmit(e) {
    e.preventDefault();
    const inp = $("askInput");
    const t = (inp.value || "").trim();
    if (!t) return;
    inp.value = "";
    $("askStatus").textContent = "Working…";
    if ($("optStream").checked) {
      runAskStream(t);
    } else {
      runAskSync(t);
    }
    setTimeout(function () {
      const st = $("askStatus");
      if (st) st.textContent = "Enter to send · Shift+Enter for newline";
    }, 400);
  }

  function newSession() {
    return fetch("/api/session/new")
      .then((r) => r.json())
      .then((d) => {
        if (d.session_id) {
          $("sessionInput").value = d.session_id;
        }
      })
      .catch(() => {});
  }

  /* init */
  document.addEventListener("DOMContentLoaded", function () {
    setTheme(getTheme());
    loadConfig();
    healthTick();
    setInterval(healthTick, 30000);

    $("tabAsk").addEventListener("click", () => setTab("ask"));
    $("tabSearch").addEventListener("click", () => setTab("search"));

    $("formSearch").addEventListener("submit", runSearch);
    $("formAsk").addEventListener("submit", onAskSubmit);
    $("askInput").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        $("formAsk").requestSubmit();
      }
    });

    $("btnTheme").addEventListener("click", () => {
      const cur = getTheme() === "light" ? "dark" : "light";
      setTheme(cur);
    });

    $("btnNewSession").addEventListener("click", (e) => {
      e.preventDefault();
      newSession();
    });

    $("btnAdvanced").addEventListener("click", () => {
      const b = $("advancedBody");
      const btn = $("btnAdvanced");
      if (!b || !btn) return;
      b.classList.toggle("hidden");
      const open = !b.classList.contains("hidden");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  });
})();
