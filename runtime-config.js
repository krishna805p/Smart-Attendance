/* Public browser configuration. Never place secrets in this file. */
(function (window) {
  "use strict";

  const isLocal = window.location.protocol === "file:" ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1";
  const localApi = "http://127.0.0.1:8000";
  // Set this public URL to the deployed AWS App Runner service before Vercel deployment.
  const productionApi = "https://YOUR-AWS-APP-RUNNER-DOMAIN";

  window.SMART_ATTENDANCE_CONFIG = Object.freeze({
    apiBaseUrl: (isLocal ? localApi : productionApi).replace(/\/$/, "")
  });
})(window);
