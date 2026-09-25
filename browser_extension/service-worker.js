const APP_URL = "https://trail-race-lab.streamlit.app/?page=itra";

async function openApp() {
  const tabs = await chrome.tabs.query({ url: "https://trail-race-lab.streamlit.app/*" });
  if (tabs.length) {
    // Keep the existing Streamlit WebSocket/session alive. Navigating the tab,
    // even to the same app with a query parameter, resets st.session_state.
    await chrome.tabs.update(tabs[0].id, { active: true });
    await chrome.windows.update(tabs[0].windowId, { focused: true });
    return tabs[0];
  }
  return chrome.tabs.create({ url: APP_URL, active: true });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "OPEN_TRAIL_RACE_LAB") return false;
  openApp()
    .then(() => sendResponse({ ok: true }))
    .catch((error) => sendResponse({ ok: false, error: String(error) }));
  return true;
});
