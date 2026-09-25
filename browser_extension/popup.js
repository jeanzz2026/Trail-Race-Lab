const button = document.querySelector("#capture");
const status = document.querySelector("#status");

function show(message, kind = "") {
  status.textContent = message;
  status.className = kind;
}

button.addEventListener("click", async () => {
  button.disabled = true;
  show("正在读取当前页面…");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !tab.url?.startsWith("https://itra.run/Races/RaceResults/")) {
      throw new Error("请先打开一个 ITRA 比赛成绩页面。");
    }
    const response = await chrome.tabs.sendMessage(tab.id, { type: "CAPTURE_ITRA_RESULTS" });
    if (!response?.ok) throw new Error(response?.error || "抓取失败");
    show(`已读取 ${response.count} 条成绩，正在打开 Trail Race Lab…`, "success");
    await chrome.runtime.sendMessage({ type: "OPEN_TRAIL_RACE_LAB" });
    window.close();
  } catch (error) {
    show(error instanceof Error ? error.message : String(error), "error");
    button.disabled = false;
  }
});
