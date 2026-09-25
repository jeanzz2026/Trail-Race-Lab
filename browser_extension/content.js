const APP_HOST = "trail-race-lab.streamlit.app";

function cleanText(element) {
  return (element?.innerText || element?.textContent || "").replace(/\s+/g, " ").trim();
}

function parseCourseInfo() {
  const text = cleanText(document.body).replace(/\u00a0/g, " ");
  const distance = text.match(/\bDistance\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:km)?\b/i);
  const gain = text.match(/\bElevation\s+Gain\s*:\s*\+?\s*([0-9][0-9,.]*)\s*(?:m)?\b/i);
  const info = {};
  if (distance) info.distance_km = Number(distance[1].replace(",", "."));
  if (gain) info.elevation_gain_m = Number(gain[1].replace(/,/g, ""));
  return info;
}

function parseResults() {
  const table = document.querySelector("#RunnerRaceResults");
  if (!table) {
    const blocked = /verify that you.re not a robot|javascript is disabled|captcha/i.test(
      cleanText(document.body),
    );
    throw new Error(
      blocked
        ? "ITRA 仍在等待人工验证。请完成验证并看到成绩表后再点击扩展。"
        : "当前页面没有找到成绩表。请确认已打开 ITRA 比赛成绩页面。",
    );
  }

  const results = [];
  for (const row of table.querySelectorAll("tr")) {
    const cells = [...row.querySelectorAll(":scope > td")];
    if (!cells.length) continue;

    const hasRaceScoreCell = cells.some((cell) => cell.hasAttribute("rowspan"));
    const ageIndex = hasRaceScoreCell ? 4 : 3;
    const genderIndex = hasRaceScoreCell ? 5 : 4;
    const nationalityIndex = hasRaceScoreCell ? 6 : 5;
    const runnerLink = cells[1]?.querySelector("a[href]");
    const nationalityText = cleanText(cells[nationalityIndex]);

    results.push({
      position: cleanText(cells[0]) || "N/A",
      name: cleanText(runnerLink) || cleanText(cells[1]) || "N/A",
      profile_link: runnerLink?.href || "N/A",
      time: cleanText(cells[2]) || "N/A",
      performance_index: "N/A",
      age: cleanText(cells[ageIndex]) || "N/A",
      gender: cleanText(cells[genderIndex]) || "N/A",
      nationality: nationalityText ? nationalityText.split(/\s+/).at(-1) : "N/A",
    });
  }

  if (!results.length) throw new Error("成绩表存在，但没有读取到有效成绩行。");
  return {
    schema_version: 1,
    capture_id: crypto.randomUUID(),
    captured_at: new Date().toISOString(),
    source_url: location.href,
    title: document.title,
    course_info: parseCourseInfo(),
    results,
  };
}

async function deliverStoredCapture() {
  if (location.hostname !== APP_HOST || window !== window.top) return;
  const { itraCapture } = await chrome.storage.local.get("itraCapture");
  if (!itraCapture) return;
  const message = { source: "trail-race-lab-extension", payload: itraCapture };
  window.postMessage(message, "*");
  for (const frame of document.querySelectorAll("iframe")) {
    frame.contentWindow?.postMessage(message, "*");
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "CAPTURE_ITRA_RESULTS") return false;
  try {
    const payload = parseResults();
    chrome.storage.local.set({ itraCapture: payload }).then(() => {
      sendResponse({ ok: true, count: payload.results.length, payload });
    });
  } catch (error) {
    sendResponse({ ok: false, error: error instanceof Error ? error.message : String(error) });
  }
  return true;
});

if (location.hostname === APP_HOST && window === window.top) {
  deliverStoredCapture();
  setInterval(deliverStoredCapture, 1000);
}
