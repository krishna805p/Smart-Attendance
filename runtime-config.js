/* Public browser configuration. Never place secrets in this file. */
(function (window) {
  "use strict";

  const isLocal = window.location.protocol === "file:" ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1";
  const localApi = "http://127.0.0.1:8000";
  // BACKEND DEPLOYMENT PENDING: replace this one public value after AWS App Runner
  // provides the real service URL. Do not put credentials or secrets here.
  const productionApi = "https://YOUR-AWS-APP-RUNNER-DOMAIN";

  window.SMART_ATTENDANCE_CONFIG = Object.freeze({
    apiBaseUrl: (isLocal ? localApi : productionApi).replace(/\/$/, "")
  });
})(window);
