// Plain React.createElement is used everywhere below (aliased to `h`) so this
// file runs as-is in the browser with no JSX transform, no Babel, and no
// build step. It works when index.html is opened directly via file://.

const h = React.createElement;
const useState = React.useState;
const useEffect = React.useEffect;
const useRef = React.useRef;
const Fragment = React.Fragment;

const API_BASE = window.SMART_ATTENDANCE_CONFIG.apiBaseUrl;

function saveAuthSession(data, role) {
  if (!data || !data.access_token) throw new Error("The server did not provide a sign-in session.");
  const user = Object.assign({}, data);
  delete user.access_token;
  delete user.token_type;
  sessionStorage.setItem("smartAttendanceToken", data.access_token);
  sessionStorage.setItem("smartAttendanceRole", role);
  sessionStorage.setItem("smartAttendanceUser", JSON.stringify(user));
  return user;
}

function clearAuthSession() {
  ["smartAttendanceToken", "smartAttendanceRole", "smartAttendanceUser"].forEach(function (key) {
    sessionStorage.removeItem(key);
  });
}

function apiRequest(path, options) {
  const requestOptions = Object.assign({}, options || {});
  requestOptions.headers = Object.assign({}, requestOptions.headers || {});
  const token = sessionStorage.getItem("smartAttendanceToken");
  if (token) requestOptions.headers.Authorization = "Bearer " + token;
  return fetch(API_BASE + path, requestOptions).then(function (response) {
    return response.json().then(function (data) {
      if (response.status === 401 && token) {
        clearAuthSession();
        window.location.replace("../index.html");
      }
      if (!response.ok) throw new Error(data.detail || "Request failed.");
      return data;
    });
  });
}

/* ------------------------------------------------------------------ */
/* GPS AND FACE-RECOGNITION FUNCTIONS                                  */
/* ------------------------------------------------------------------ */

function loginStudent(rollNo, password) {
  return apiRequest("/api/student/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roll_no: rollNo, password: password })
  }).then(function (student) {
    return saveAuthSession(student, "student");
  });
}

function verifyLocation() {
  return new Promise(function (resolve, reject) {
    if (!navigator.geolocation) {
      reject(new Error("Location services are not supported by this browser."));
      return;
    }

    navigator.geolocation.getCurrentPosition(
      function (position) {
        apiRequest("/api/verify-location", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          })
        }).then(function (result) {
          const distance = result.distance_meters.toFixed(1) + "m";
          const radius = result.radius_meters + "m";
          resolve({
            success: result.inside_geofence,
            location: result.inside_geofence ? "Within Block 33 attendance zone" : "Outside Block 33 attendance zone",
            distance: "Distance from Block 33: " + distance + " · Allowed radius: " + radius
          });
        }).catch(reject);
      },
      function (error) {
        const messages = {
          1: "Location permission was denied. Allow location access and try again.",
          2: "Your location is currently unavailable. Check location services and try again.",
          3: "Location request timed out. Try again in an area with better signal."
        };
        reject(new Error(messages[error.code] || "Unable to retrieve your location."));
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  });
}

/* ------------------------------------------------------------------ */
/* SMALL SHARED COMPONENTS                                             */
/* ------------------------------------------------------------------ */

function StatusPill(props) {
  const status = props.status;
  const cls = status === "Present" ? "present" : status === "Absent" ? "absent" : "late";
  return h("span", { className: "pill " + cls },
    h("span", { className: "dot" }),
    status
  );
}

function TopBar(props) {
  const student = props.student;
  const initials = student.name.split(" ").map((n) => n[0]).slice(0, 2).join("");
  return h("div", { className: "topbar" },
    h("div", { className: "topbar-inner" },
      h("div", { className: "topbar-user" },
        h("div", { className: "avatar" }, initials),
        h("div", null,
          h("div", { className: "topbar-name" }, student.name),
          h("div", { className: "topbar-roll" }, student.rollNo)
        )
      ),
      h("button", { className: "logout-btn", onClick: props.onLogout },
        h("span", null, "\u23FB"),
        h("span", { className: "label" }, "Logout")
      )
    )
  );
}

function FlowSteps(props) {
  const step = props.step;
  const steps = ["Location", "Face scan", "Result"];
  const children = [];
  steps.forEach((label, i) => {
    children.push(
      h("div", { className: "flow-step", key: "step-" + i },
        h("div", { className: "flow-step-num " + (i < step ? "done" : i === step ? "active" : "") },
          i < step ? "\u2713" : String(i + 1)
        ),
        h("span", { className: "flow-step-label " + (i === step ? "active" : "") }, label)
      )
    );
    if (i < steps.length - 1) {
      children.push(h("div", { className: "flow-connector " + (i < step ? "done" : ""), key: "conn-" + i }));
    }
  });
  return h("div", { className: "flow-steps" }, children);
}

/* ------------------------------------------------------------------ */
/* LOGIN SCREEN                                                         */
/* ------------------------------------------------------------------ */

function LoginScreen(props) {
  const rollNoState = useState("");
  const rollNo = rollNoState[0], setRollNo = rollNoState[1];
  const passwordState = useState("");
  const password = passwordState[0], setPassword = passwordState[1];
  const loadingState = useState(false);
  const loading = loadingState[0], setLoading = loadingState[1];
  const errorState = useState("");
  const error = errorState[0], setError = errorState[1];

  function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    loginStudent(rollNo, password)
      .then(function (student) { props.onLogin(student); })
      .catch(function (err) { setError(err.message); })
      .finally(function () { setLoading(false); });
  }

  return h("div", { className: "centered-screen login-bg" },
    h("div", { className: "login-glow-1" }),
    h("div", { className: "login-glow-2" }),
    h("div", { className: "login-wrap" },
      h("div", { className: "login-brand" },
        h("div", { className: "login-icon" }, "\uD83E\uDEAA"),
        h("h1", null, "Smart Attendance"),
        h("p", null, "Face + GPS verified check-in")
      ),
      h("div", { className: "login-card" },
        h("h2", null, "Student login"),
        h("form", { onSubmit: handleSubmit },
          h("div", { className: "field" },
            h("label", null, "Roll number"),
            h("div", { className: "field-input-wrap" },
              h("span", { className: "field-icon" }, "\uD83C\uDF93"),
              h("input", {
                type: "text",
                placeholder: "e.g. CT21BBA045",
                value: rollNo,
                onChange: function (e) { setRollNo(e.target.value); }
              })
            )
          ),
          h("div", { className: "field" },
            h("label", null, "Password"),
            h("div", { className: "field-input-wrap" },
              h("span", { className: "field-icon" }, "\uD83D\uDD12"),
              h("input", {
                type: "password",
                placeholder: "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022",
                value: password,
                onChange: function (e) { setPassword(e.target.value); }
              })
            )
          ),
          error ? h("div", { className: "error-box" }, "\u26A0 " + error) : null,
          h("button", { className: "btn-primary", type: "submit", disabled: loading },
            loading ? "Signing in..." : "Sign in"
          )
        )
      ),
      h("p", { className: "login-hint" }, "Use your registered roll number and password.")
    )
  );
}

/* ------------------------------------------------------------------ */
/* DASHBOARD SCREEN                                                     */
/* ------------------------------------------------------------------ */

function Dashboard(props) {
  const student = props.student;
  const history = props.history;
  const meetsRequirement = student.attendancePercent >= 75;
  const initials = student.name.split(" ").map((n) => n[0]).slice(0, 2).join("");

  return h("div", { className: "screen" },
    h(TopBar, { student: student, onLogout: props.onLogout }),
    h("div", { className: "app-content" },
      h("div", { className: "greeting" },
        h("h1", null, "Welcome back, " + student.name.split(" ")[0] + " \uD83D\uDC4B"),
        h("p", null, "Here's your attendance overview for today.")
      ),

      h("div", { className: "card" },
        h("div", { className: "profile-head" },
          h("div", { className: "profile-head-left" },
            h("div", { className: "profile-avatar" }, initials),
            h("div", null,
              h("h2", null, student.name),
              h("p", null, student.rollNo + " \u00B7 " + student.course)
            )
          ),
          h("span", { className: "status-badge " + (meetsRequirement ? "ok" : "warn") },
            "\uD83D\uDEE1 " + (meetsRequirement ? "On track" : "Below requirement")
          )
        ),

        h("div", { className: "stat-grid" },
          h("div", { className: "stat-tile indigo" },
            h("div", { className: "num" }, student.attendancePercent + "%"),
            h("div", { className: "label" }, "Attendance")
          ),
          h("div", { className: "stat-tile emerald" },
            h("div", { className: "num" }, String(student.present)),
            h("div", { className: "label" }, "Present")
          ),
          h("div", { className: "stat-tile slate" },
            h("div", { className: "num" }, String(student.totalClasses)),
            h("div", { className: "label" }, "Total classes")
          )
        ),

        h("div", { className: "progress-row" },
          h("span", null, "Attendance progress"),
          h("strong", null, student.attendancePercent + "% \u00B7 75% required")
        ),
        h("div", { className: "progress-track" },
          h("div", {
            className: "progress-fill " + (meetsRequirement ? "ok" : "warn"),
            style: { width: student.attendancePercent + "%" }
          })
        )
      ),

      h("button", { className: "cta-button", onClick: props.onStartAttendance },
        h("div", { className: "cta-left" },
          h("div", { className: "cta-icon" }, "\uD83D\uDCF7"),
          h("div", null,
            h("div", { className: "cta-title" }, "Mark today's attendance"),
            h("div", { className: "cta-sub" }, "Requires location + face verification")
          )
        ),
        h("span", null, "\u203A")
      ),

      h("div", { className: "card" },
        h("div", { className: "section-head" },
          h("h3", null, "\uD83D\uDD52 Recent activity"),
          h("button", { className: "link-btn", onClick: props.onViewHistory }, "View all \u203A")
        ),
        h("div", null,
          history.slice(0, 3).map(function (hist, i) {
            return h("div", { className: "history-row", key: i },
              h("div", null,
                h("div", { className: "history-date" }, hist.date),
                h("div", { className: "history-time" }, hist.time)
              ),
              h(StatusPill, { status: hist.status })
            );
          })
        )
      )
    )
  );
}

/* ------------------------------------------------------------------ */
/* GPS VERIFICATION STEP                                                */
/* ------------------------------------------------------------------ */

function GPSStep(props) {
  const statusState = useState("checking"); // checking | success | fail
  const status = statusState[0], setStatus = statusState[1];
  const detailState = useState(null);
  const detail = detailState[0], setDetail = detailState[1];

  function runCheck() {
    setStatus("checking");
    setDetail(null);
    verifyLocation()
      .then(function (res) {
        setDetail(res);
        setStatus(res.success ? "success" : "fail");
      })
      .catch(function (error) {
        setDetail({ location: error.message, distance: "" });
        setStatus("fail");
      });
  }

  useEffect(function () {
    let mounted = true;
    verifyLocation()
      .then(function (res) {
        if (!mounted) return;
        setDetail(res);
        setStatus(res.success ? "success" : "fail");
      })
      .catch(function (error) {
        if (!mounted) return;
        setDetail({ location: error.message, distance: "" });
        setStatus("fail");
      });
    return function () { mounted = false; };
  }, []);

  useEffect(function () {
    if (status === "success") {
      const t = setTimeout(function () { props.onComplete(detail); }, 700);
      return function () { clearTimeout(t); };
    }
  }, [status]);

  return h("div", { className: "centered-screen", style: { flexDirection: "column" } },
    h(FlowSteps, { step: 0 }),
    h("div", { className: "flow-card" },
      h("div", { className: "flow-icon-circle " + status },
        status === "checking" ? h("span", { style: { position: "relative" } },
          "\uD83D\uDCCD",
          h("span", { className: "pulse-ring" })
        ) : null,
        status === "success" ? "\u2705" : null,
        status === "fail" ? "\u274C" : null
      ),
      h("p", { className: "flow-title" },
        status === "checking" ? "Verifying your location..." :
        status === "success" ? "Location verified" : "Location check failed"
      ),
      h("p", { className: "flow-sub" }, "\uD83D\uDCCD " + (detail ? detail.location : "Fetching GPS coordinates")),
      detail ? h("p", { className: "flow-detail" }, detail.distance) : h("div", { style: { marginBottom: 20 } }),

      status === "checking" ? h("div", { className: "loading-bar-track" },
        h("div", { className: "loading-bar-fill" })
      ) : null,

      status === "fail" ? h(Fragment, null,
        h("button", { className: "btn-primary", onClick: runCheck, style: { marginBottom: 8 } }, "Retry location check"),
        h("button", { className: "btn-secondary", onClick: props.onCancel }, "Cancel")
      ) : null
    )
  );
}

/* ------------------------------------------------------------------ */
/* FACE SCAN STEP                                                       */
/* ------------------------------------------------------------------ */

function FaceScanStep(props) {
  const statusState = useState("scanning"); // scanning | success | fail
  const status = statusState[0], setStatus = statusState[1];
  const resultState = useState(null);
  const result = resultState[0], setResult = resultState[1];
  const cameraReadyState = useState(false);
  const cameraReady = cameraReadyState[0], setCameraReady = cameraReadyState[1];
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const scanInProgressRef = useRef(false);

  function cameraErrorMessage(error) {
    if (error instanceof TypeError) {
      return "Could not reach the face-verification service. Check your connection and try again.";
    }
    if (error && (error.name === "NotAllowedError" || error.name === "SecurityError")) {
      return "Camera permission was denied. Allow camera access and try again.";
    }
    if (error && error.name === "NotFoundError") return "No webcam was found.";
    if (error && error.name === "NotReadableError") return "The webcam is unavailable or being used by another app.";
    return (error && error.message) || "Unable to access the webcam.";
  }

  function waitForVideoFrame(video) {
    return new Promise(function (resolve, reject) {
      let attempts = 0;
      function check() {
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          resolve();
        } else if (attempts++ >= 45) {
          reject(new Error("The webcam started but did not provide a video frame."));
        } else {
          requestAnimationFrame(check);
        }
      }
      check();
    });
  }

  function captureWebcamBlob(video) {
    const maxDimension = 1280;
    const scale = Math.min(1, maxDimension / video.videoWidth, maxDimension / video.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) return Promise.reject(new Error("Could not capture a webcam frame."));
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise(function (resolve, reject) {
      canvas.toBlob(function (blob) {
        if (blob) resolve(blob);
        else reject(new Error("Could not encode the webcam frame."));
      }, "image/jpeg", 0.85);
    });
  }

  function runScan() {
    if (scanInProgressRef.current) return;
    scanInProgressRef.current = true;
    setStatus("scanning");
    setResult(null);
    const camera = streamRef.current
      ? Promise.resolve(streamRef.current)
      : (navigator.mediaDevices && navigator.mediaDevices.getUserMedia
          ? navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
          : Promise.reject(new Error("This browser does not support webcam access.")));

    camera.then(function (stream) {
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("The webcam preview is unavailable.");
      video.srcObject = stream;
      setCameraReady(true);
      return video.play().then(function () { return video; });
    }).then(function (video) {
      return waitForVideoFrame(video).then(function () { return captureWebcamBlob(video); });
    }).then(function (blob) {
      const form = new FormData();
      form.append("image", blob, "webcam.jpg");
      return apiRequest("/api/student/verify-face", { method: "POST", body: form });
    }).then(function (res) {
      setResult(res);
      setStatus(res.verified ? "success" : "fail");
    }).catch(function (error) {
      setCameraReady(false);
      setResult({ message: cameraErrorMessage(error) });
      setStatus("fail");
    }).finally(function () {
      scanInProgressRef.current = false;
    });
  }

  useEffect(function () {
    runScan();

    return function () {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(function (t) { t.stop(); });
      }
    };
  }, []);

  const bracketClass = status === "success" ? "success" : status === "fail" ? "fail" : "";

  return h("div", { className: "centered-screen", style: { flexDirection: "column" } },
    h(FlowSteps, { step: 1 }),
    h("div", { className: "flow-card" },
      h("p", { className: "flow-title", style: { marginBottom: 16 } }, "Face verification"),

      h("div", { className: "face-frame" },
        h("video", { ref: videoRef, autoPlay: true, muted: true, playsInline: true,
          style: { display: cameraReady ? "block" : "none" } }),
        !cameraReady ? h("div", { className: "face-frame-placeholder" },
              h("span", { className: "icon" }, "\uD83D\uDCF7"),
              h("span", null, "Camera preview unavailable")
            ) : null,
        h("div", { className: "face-brackets " + bracketClass }),
        status === "scanning" ? h("div", { className: "scan-line" }) : null
      ),

      h("div", { className: "face-status" },
        status === "scanning" ? "Capturing and verifying face..." :
        status === "success" ? (result && result.threshold_calibrated
          ? "\u2705 Face matched" : "\u26A0\uFE0F Provisional face match") : "\u274C Face verification failed"
      ),
      result ? h("div", { className: "face-confidence" },
        result.score !== undefined
          ? "Similarity: " + Number(result.score).toFixed(4) + " · Threshold: " +
            Number(result.threshold).toFixed(2) + (result.threshold_calibrated ? "" : " (not calibrated)")
          : (result.message || "Face verification failed."),
        result.message ? h("div", { style: { marginTop: 6 } }, result.message) : null
      ) : h("div", { style: { marginBottom: 20 } }),

      status === "fail" ? h(Fragment, null,
        h("button", { className: "btn-primary", onClick: runScan, style: { marginBottom: 8 } }, "Try again"),
        h("button", { className: "btn-secondary", onClick: props.onCancel }, "Cancel")
      ) : status === "success" ? h("button", {
        className: "btn-secondary", onClick: props.onCancel
      }, "Return to dashboard (attendance not submitted)") : null
    )
  );
}

/* ------------------------------------------------------------------ */
/* RESULT STEP                                                          */
/* ------------------------------------------------------------------ */

function ResultStep(props) {
  const success = props.success;
  return h("div", { className: "centered-screen", style: { flexDirection: "column" } },
    h(FlowSteps, { step: 2 }),
    h("div", { className: "flow-card" },
      h("div", { className: "result-icon-circle " + (success ? "success" : "fail") },
        success ? "\u2705" : "\u274C"
      ),
      h("p", { className: "result-title" }, success ? "Attendance marked" : "Attendance failed"),
      h("p", { className: "result-sub" },
        success
          ? "Your attendance has been recorded successfully."
          : "We couldn't verify your identity or location. Please try again."
      ),
      success ? h("div", { className: "result-tag" }, "\u2728 Face + GPS verified") : null,
      h("button", { className: "btn-primary", onClick: props.onDone }, "Back to dashboard")
    )
  );
}

/* ------------------------------------------------------------------ */
/* HISTORY SCREEN                                                       */
/* ------------------------------------------------------------------ */

function HistoryScreen(props) {
  const student = props.student;
  const history = props.history;
  const presentCount = history.filter(function (hh) { return hh.status === "Present"; }).length;
  const absentCount = history.filter(function (hh) { return hh.status === "Absent"; }).length;
  const lateCount = history.filter(function (hh) { return hh.status === "Late"; }).length;

  return h("div", { className: "screen" },
    h(TopBar, { student: student, onLogout: props.onLogout }),
    h("div", { className: "app-content" },
      h("button", { className: "back-link", onClick: props.onBack }, "\u2039 Back to dashboard"),
      h("h1", { className: "page-title" }, "Attendance history"),

      h("div", { className: "stat-mini-grid" },
        h("div", { className: "stat-mini" },
          h("div", { className: "num", style: { color: "var(--emerald)" } }, String(presentCount)),
          h("div", { className: "label" }, "Present")
        ),
        h("div", { className: "stat-mini" },
          h("div", { className: "num", style: { color: "var(--rose)" } }, String(absentCount)),
          h("div", { className: "label" }, "Absent")
        ),
        h("div", { className: "stat-mini" },
          h("div", { className: "num", style: { color: "var(--amber)" } }, String(lateCount)),
          h("div", { className: "label" }, "Late")
        )
      ),

      h("div", { className: "card", style: { padding: 0 } },
        h("div", {
          className: "section-head",
          style: { padding: "16px 20px", margin: 0, borderBottom: "1px solid var(--slate-100)" }
        },
          h("h3", null, "\uD83D\uDCC8 Full record"),
          h("span", { style: { fontSize: 13.5, fontWeight: 600, color: "var(--indigo)" } },
            student.attendancePercent + "% overall"
          )
        ),

        h("div", { className: "history-table-wrap" },
          h("table", { className: "history-table" },
            h("thead", null,
              h("tr", null,
                h("th", null, "Date"),
                h("th", null, "Time"),
                h("th", null, "Method"),
                h("th", null, "Status")
              )
            ),
            h("tbody", null,
              history.map(function (hist, i) {
                return h("tr", { key: i },
                  h("td", { className: "date-cell" }, hist.date),
                  h("td", null, hist.time),
                  h("td", null, hist.method),
                  h("td", null, h(StatusPill, { status: hist.status }))
                );
              })
            )
          )
        ),

        h("div", { className: "mobile-history-list" },
          history.map(function (hist, i) {
            return h("div", { className: "mobile-history-row", key: i },
              h("div", null,
                h("div", { className: "m-date" }, hist.date),
                h("div", { className: "m-meta" }, hist.time + " \u00B7 " + hist.method)
              ),
              h(StatusPill, { status: hist.status })
            );
          })
        )
      )
    )
  );
}

/* ------------------------------------------------------------------ */
/* ROOT APP                                                             */
/* ------------------------------------------------------------------ */

function App() {
  const screenState = useState("login"); // login | dashboard | gps | face | result | history
  const screen = screenState[0], setScreen = screenState[1];
  const studentState = useState(null);
  const student = studentState[0], setStudent = studentState[1];
  const historyState = useState([]);
  const history = historyState[0], setHistory = historyState[1];
  const lastResultState = useState(true);
  const lastResultSuccess = lastResultState[0], setLastResultSuccess = lastResultState[1];

  function handleLogin(data) {
    return Promise.all([
      apiRequest("/api/student/" + encodeURIComponent(data.rollNo)),
      apiRequest("/api/student/" + encodeURIComponent(data.rollNo) + "/attendance")
    ]).then(function (results) {
      setStudent(results[0]);
      setHistory(results[1]);
      setScreen("dashboard");
    });
  }

  function handleLogout() {
    clearAuthSession();
    window.location.replace("../index.html");
  }

  useEffect(function () {
    const role = sessionStorage.getItem("smartAttendanceRole");
    const token = sessionStorage.getItem("smartAttendanceToken");
    if (!token) return;
    if (role !== "student") {
      window.location.replace("../index.html");
      return;
    }
    try {
      const user = JSON.parse(sessionStorage.getItem("smartAttendanceUser") || "{}");
      if (!user.rollNo) throw new Error("Missing student session identity.");
      handleLogin(user).catch(function () {
        clearAuthSession();
        setStudent(null);
        setScreen("login");
      });
    } catch (_error) {
      clearAuthSession();
      setScreen("login");
    }
  }, []);

  function handleGPSComplete() {
    setScreen("face");
  }

  if (screen === "login") return h(LoginScreen, { onLogin: handleLogin });

  if (screen === "dashboard") return h(Dashboard, {
    student: student,
    history: history,
    onStartAttendance: function () { setScreen("gps"); },
    onViewHistory: function () { setScreen("history"); },
    onLogout: handleLogout
  });

  if (screen === "gps") return h(GPSStep, {
    onComplete: handleGPSComplete,
    onCancel: function () { setScreen("dashboard"); }
  });

  if (screen === "face") return h(FaceScanStep, {
    onCancel: function () { setScreen("dashboard"); }
  });

  if (screen === "result") return h(ResultStep, {
    success: lastResultSuccess,
    onDone: function () { setScreen("dashboard"); }
  });

  if (screen === "history") return h(HistoryScreen, {
    student: student,
    history: history,
    onBack: function () { setScreen("dashboard"); },
    onLogout: handleLogout
  });

  return null;
}

const rootEl = document.getElementById("root");
const root = ReactDOM.createRoot(rootEl);
root.render(h(App));
