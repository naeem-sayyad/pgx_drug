(() => {
  const fileInput = document.getElementById("dnaFile");
  const drugField = document.getElementById("drugField");
  const analyzeButton = document.getElementById("analyzeButton");
  const statusPanel = document.getElementById("statusPanel");
  const resultsPanel = document.getElementById("resultsPanel");
  const statusText = document.getElementById("statusText");
  const countdownText = document.getElementById("countdownText");
  const summaryList = document.getElementById("summaryList");
  const downloadCsvButton = document.getElementById("downloadCsvButton");
  const chatbotToggle = document.getElementById("chatbotToggle");
  const chatbotPanel = document.getElementById("chatbotPanel");
  const chatbotClose = document.getElementById("chatbotClose");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatSubmit = chatForm ? chatForm.querySelector("button") : null;
  const chatLog = document.getElementById("chatLog");
  const idleVideo = document.getElementById("idleVideo");
  const analysisVideo = document.getElementById("analysisVideo");

  let analysisData = null;
  let downloadUrl = "";
  let currentAnalysisId = null;

  if (analyzeButton) {
    analyzeButton.disabled = true;
  }
  function safelyPlay(video) {
    if (!video) {
      return;
    }
    const playback = video.play();
    if (playback && typeof playback.catch === "function") {
      playback.catch(() => {});
    }
  }

  function activateVideo(target) {
    if (!idleVideo || !analysisVideo) {
      return;
    }
    if (target === "analysis") {
      idleVideo.classList.remove("active");
      analysisVideo.classList.add("active");
      safelyPlay(analysisVideo);
    } else {
      analysisVideo.classList.remove("active");
      idleVideo.classList.add("active");
      safelyPlay(idleVideo);
    }
  }

  function setChatEnabled(enabled) {
    if (!chatInput || !chatSubmit) {
      return;
    }
    chatInput.disabled = !enabled;
    chatSubmit.disabled = !enabled;
    chatInput.placeholder = enabled
      ? "Ask about toxicity, PMIDs, or clinical evidence…"
      : "Type your question...";
  }

  function appendChatMessage(role, text, citations = []) {
    if (!chatLog) {
      return;
    }
    const container = document.createElement("div");
    container.className = `chat-message ${role}`;

    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    container.appendChild(paragraph);

    if (Array.isArray(citations) && citations.length > 0) {
      const citationList = document.createElement("ul");
      citationList.className = "citation-list";
      citations.forEach((entry) => {
        const item = document.createElement("li");
        const pmidText = entry.pmid ? `PMID ${entry.pmid}` : "Evidence";
        const titleText = entry.title || pmidText;
        if (entry.url) {
          const link = document.createElement("a");
          link.href = entry.url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = titleText;
          item.appendChild(link);
        } else {
          item.textContent = titleText;
        }
        const metaParts = [];
        if (entry.journal) {
          metaParts.push(entry.journal);
        }
        if (entry.year) {
          metaParts.push(entry.year);
        }
        if (metaParts.length > 0) {
          const metaSpan = document.createElement("span");
          metaSpan.className = "reference-meta";
          metaSpan.textContent = metaParts.join(" · ");
          item.appendChild(metaSpan);
        }
        const pmidBadge = document.createElement("span");
        pmidBadge.className = "pmid-badge";
        pmidBadge.textContent = pmidText;
        item.appendChild(pmidBadge);

        citationList.appendChild(item);
      });
      container.appendChild(citationList);
    }

    chatLog.appendChild(container);
    chatLog.scrollTop = chatLog.scrollHeight;
    chatLog.dataset.active = "1";
  }

  function resetState() {
    statusPanel.classList.add("hidden");
    resultsPanel.classList.add("hidden");
    countdownText.textContent = "";
    summaryList.innerHTML = "";
    downloadCsvButton.disabled = true;
    downloadUrl = "";
    analysisData = null;
    currentAnalysisId = null;
    activateVideo("idle");
    setChatEnabled(false);
    if (chatLog) {
      chatLog.dataset.active = "0";
      chatLog.innerHTML = '<p class="bot-message">Upload a DNA file to begin exploring the ensemble evidence.</p>';
    }
    if (chatInput) {
      chatInput.value = "";
    }
    if (drugField) {
      drugField.classList.remove("hidden");
      drugField.classList.add("visible");
    }
    if (analyzeButton) {
      analyzeButton.disabled = !fileInput.files || fileInput.files.length === 0;
      analyzeButton.classList.remove("hidden");
    }
  }

  resetState();

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      if (drugField) {
        drugField.classList.remove("hidden");
        requestAnimationFrame(() => {
          drugField.classList.add("visible");
        });
      }
      if (analyzeButton) {
        analyzeButton.disabled = false;
      }
    } else if (analyzeButton) {
      analyzeButton.disabled = true;
    }
  });

  function renderResults(data) {
    statusPanel.classList.add("hidden");
    summaryList.innerHTML = "";
    (data.summaries || []).forEach((entry) => {
      const card = document.createElement("article");
      card.className = "summary-card";

      const title = document.createElement("h3");
      title.textContent = `${entry.gene} × ${entry.drug}`;
      card.appendChild(title);

      const probability = document.createElement("div");
      probability.className = "model-metrics";
      probability.innerHTML = `
        <span class="metric">Ensemble: ${Number(entry.ensemble_probability || 0).toFixed(2)}%</span>
        <span class="metric">XGB: ${Number(entry.probabilities.xgb || 0).toFixed(2)}%</span>
        <span class="metric">CatBoost: ${Number(entry.probabilities.catboost || 0).toFixed(2)}%</span>
        <span class="metric">TargetTox: ${Number(entry.probabilities.targettox || 0).toFixed(2)}%</span>
      `;
      card.appendChild(probability);

      const summary = document.createElement("p");
      summary.textContent = entry.summary;
      card.appendChild(summary);

      const evidencePanel = document.createElement("div");
      evidencePanel.className = "evidence-panel";

      const evidenceHeading = document.createElement("h4");
      evidenceHeading.textContent = "Scientific Evidence";
      evidencePanel.appendChild(evidenceHeading);

      const cards = entry.evidence_details || [];
      if (cards.length === 0) {
        const fallback = document.createElement("p");
        fallback.className = "evidence-fallback";
        fallback.textContent =
          entry.evidence || "Evidence sourced from curated pharmacogenomic knowledge bases.";
        evidencePanel.appendChild(fallback);
      } else {
        cards.forEach((detail) => {
          const evidenceCard = document.createElement("article");
          evidenceCard.className = "evidence-card";

          const heading = document.createElement("div");
          heading.className = "evidence-card__heading";
          heading.textContent = detail.headline || "Clinical evidence";
          evidenceCard.appendChild(heading);

          const meta = document.createElement("div");
          meta.className = "evidence-card__meta";
          if (detail.strength) {
            const strength = document.createElement("span");
            strength.textContent = `Strength: ${detail.strength}`;
            meta.appendChild(strength);
          }
          if (detail.significance) {
            const significance = document.createElement("span");
            significance.textContent = `Significance: ${detail.significance}`;
            meta.appendChild(significance);
          }
          if (detail.journal) {
            const journal = document.createElement("span");
            journal.textContent = detail.year
              ? `${detail.journal} (${detail.year})`
              : detail.journal;
            meta.appendChild(journal);
          } else if (detail.year) {
            const year = document.createElement("span");
            year.textContent = detail.year;
            meta.appendChild(year);
          }
          if (meta.children.length > 0) {
            evidenceCard.appendChild(meta);
          }

          if (detail.title) {
            const titleElem = document.createElement("p");
            titleElem.className = "evidence-card__title";
            titleElem.textContent = detail.year
              ? `${detail.title} (${detail.year})`
              : detail.title;
            evidenceCard.appendChild(titleElem);
          }

          const snippet = document.createElement("p");
          snippet.className = "evidence-card__snippet";
          snippet.textContent = detail.snippet;
          evidenceCard.appendChild(snippet);

          if (detail.notes && detail.notes !== detail.snippet) {
            const notes = document.createElement("p");
            notes.className = "evidence-card__notes";
            notes.textContent = detail.notes;
            evidenceCard.appendChild(notes);
          }

          const citationDetails = detail.citations || [];
          if (citationDetails.length > 0) {
            const references = document.createElement("div");
            references.className = "evidence-card__references";

            const heading = document.createElement("span");
            heading.textContent = citationDetails.length > 1 ? "References" : "Reference";
            references.appendChild(heading);

            const list = document.createElement("ul");
            list.className = "reference-list";
            citationDetails.forEach((cite) => {
              if (!cite.pmid) {
                return;
              }
              const item = document.createElement("li");
              const titleText = cite.title || `PMID ${cite.pmid}`;
              if (cite.url) {
                const link = document.createElement("a");
                link.href = cite.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = titleText;
                item.appendChild(link);
              } else {
                item.textContent = titleText;
              }
              const metaParts = [];
              if (cite.journal) {
                metaParts.push(cite.journal);
              }
              if (cite.year) {
                metaParts.push(cite.year);
              }
              const metaText = metaParts.join(" · ");
              if (metaText) {
                const metaSpan = document.createElement("span");
                metaSpan.className = "reference-meta";
                metaSpan.textContent = metaText;
                item.appendChild(metaSpan);
              }
              const pmidBadge = document.createElement("span");
              pmidBadge.className = "pmid-badge";
              pmidBadge.textContent = `PMID ${cite.pmid}`;
              item.appendChild(pmidBadge);

              list.appendChild(item);
            });
            references.appendChild(list);
            evidenceCard.appendChild(references);
          } else if (Array.isArray(detail.pmids) && detail.pmids.length > 0) {
            const citationRow = document.createElement("div");
            citationRow.className = "evidence-card__citations";
            citationRow.textContent = `PMIDs: ${detail.pmids.join(", ")}`;
            evidenceCard.appendChild(citationRow);
          }

          evidencePanel.appendChild(evidenceCard);
        });
      }

      card.appendChild(evidencePanel);
      summaryList.appendChild(card);
    });

    downloadUrl = `/download/${data.analysis_id}`;
    downloadCsvButton.disabled = false;
    resultsPanel.classList.remove("hidden");
    currentAnalysisId = data.analysis_id || null;
    setChatEnabled(Boolean(currentAnalysisId));
    if (chatLog && chatLog.dataset.active !== "1") {
      chatLog.innerHTML = "";
      appendChatMessage(
        "bot",
        "Results ready. Ask about specific genes, drugs, or request the supporting PMIDs to dive deeper."
      );
    }
    activateVideo("idle");
  }

  document
    .getElementById("analysisForm")
    .addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please choose your DNA file first.");
        return;
      }

      resetState();
      statusPanel.classList.remove("hidden");
      statusText.textContent = "Running Advanced DNA Analysis on this Gene.";
      activateVideo("analysis");
      countdownText.textContent = "Analyzing sample…";

      const formData = new FormData(event.target);

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || "Failed to analyze DNA.");
        }

        analysisData = await response.json();
        renderResults(analysisData);
      } catch (error) {
        resetState();
        alert(error.message || "Something went wrong while running the analysis.");
      }
    });

  downloadCsvButton.addEventListener("click", () => {
    if (!downloadUrl) {
      return;
    }
    window.open(downloadUrl, "_blank");
  });

  chatbotToggle.addEventListener("click", () => {
    chatbotPanel.classList.toggle("hidden");
    if (!chatbotPanel.classList.contains("hidden") && chatInput && !chatInput.disabled) {
      chatInput.focus();
    }
  });

  chatbotClose.addEventListener("click", () => {
    chatbotPanel.classList.add("hidden");
  });

  if (chatForm) {
    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!chatInput || chatInput.disabled) {
        return;
      }
      const message = chatInput.value.trim();
      if (!message) {
        return;
      }
      if (!currentAnalysisId) {
        appendChatMessage(
          "bot",
          "Run an analysis first so I can ground the conversation in the latest evidence."
        );
        return;
      }
      appendChatMessage("user", message);
      chatInput.value = "";
      setChatEnabled(false);
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            analysis_id: currentAnalysisId,
            message,
          }),
        });
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || "Unable to retrieve a response.");
        }
        const payload = await response.json();
        appendChatMessage(
          "bot",
          payload.reply || "I could not find evidence for that request.",
          payload.citations
        );
      } catch (error) {
        appendChatMessage(
          "bot",
          error.message || "Something went wrong while consulting the evidence."
        );
      } finally {
        setChatEnabled(true);
        if (chatInput && !chatInput.disabled) {
          chatInput.focus();
        }
      }
    });
  }
})();
