// Plain React.createElement is used everywhere below (aliased to `h`) so this
// file runs as-is in the browser with no JSX transform, no Babel, and no
// build step — works via file:// or VS Code Live Server identically.

const h = React.createElement;
const useState = React.useState;
const useMemo = React.useMemo;
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

function loginTeacher(username, password) {
  return apiRequest("/api/teacher/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, password: password })
  }).then(function (teacher) {
    return saveAuthSession(teacher, "teacher");
  });
}

function toDateStr(d) {
  return d.toISOString().slice(0, 10);
}

function daysAgo(n) {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return toDateStr(d);
}

const TODAY = toDateStr(new Date());
let teacherAttendancePollInFlight = false;

function uniqueAttendanceRows(rows) {
  const byStudent = new Map();
  (rows || []).forEach(function (row) {
    if (!byStudent.has(row.studentId)) byStudent.set(row.studentId, row);
  });
  return Array.from(byStudent.values());
}

function sameAttendanceRows(first, second) {
  if (first === second) return true;
  if (!first || !second || first.length !== second.length) return false;
  for (let i = 0; i < first.length; i += 1) {
    const a = first[i], b = second[i];
    if (a.id !== b.id || a.studentId !== b.studentId || a.rollNo !== b.rollNo ||
        a.date !== b.date || a.time !== b.time || a.status !== b.status || a.method !== b.method) {
      return false;
    }
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* HELPERS                                                              */
/* ------------------------------------------------------------------ */

function studentById(students, id) {
  return students.find(function (s) { return s.id === id; });
}

function downloadCSV(rows, students, filename) {
  const header = ["Roll number", "Name", "Date", "Time", "Status", "Method"];
  const lines = [header.join(",")];
  rows.forEach(function (r) {
    const student = studentById(students, r.studentId);
    const line = [
      student ? student.rollNo : "",
      student ? student.name : "",
      r.date,
      r.time || "--",
      r.status,
      r.method || "--",
    ].map(function (v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(",");
    lines.push(line);
  });
  const csvContent = lines.join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
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
  const teacher = props.teacher;
  const initials = teacher.name.replace("Dr. ", "").split(" ").map(function (n) { return n[0]; }).slice(0, 2).join("");
  return h("div", { className: "topbar" },
    h("div", { className: "topbar-inner" },
      h("div", { className: "topbar-user" },
        h("div", { className: "avatar" }, initials),
        h("div", null,
          h("div", { className: "topbar-name" }, teacher.name),
          h("div", { className: "topbar-role" }, teacher.role)
        )
      ),
      h("button", { className: "logout-btn", onClick: props.onLogout },
        h("span", null, "\u23FB"),
        h("span", { className: "label" }, "Logout")
      )
    )
  );
}

/* ------------------------------------------------------------------ */
/* LOGIN SCREEN                                                         */
/* ------------------------------------------------------------------ */

function LoginScreen(props) {
  const usernameState = useState("");
  const username = usernameState[0], setUsername = usernameState[1];
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
    loginTeacher(username, password)
      .then(function (teacher) { props.onLogin(teacher); })
      .catch(function (err) { setError(err.message); })
      .finally(function () { setLoading(false); });
  }

  return h("div", { className: "centered-screen login-bg" },
    h("div", { className: "login-glow-1" }),
    h("div", { className: "login-glow-2" }),
    h("div", { className: "login-wrap" },
      h("div", { className: "login-brand" },
        h("div", { className: "login-icon" }, "\uD83C\uDF93"),
        h("h1", null, "Smart Attendance"),
        h("p", null, "Faculty portal")
      ),
      h("div", { className: "login-card" },
        h("h2", null, "Faculty login"),
        h("form", { onSubmit: handleSubmit },
          h("div", { className: "field" },
            h("label", null, "Username"),
            h("div", { className: "field-input-wrap" },
              h("span", { className: "field-icon" }, "\uD83D\uDC64"),
              h("input", {
                type: "text",
                placeholder: "e.g. akapoor",
                value: username,
                onChange: function (e) { setUsername(e.target.value); }
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
      h("p", { className: "login-hint" }, "Use your faculty username and password.")
    )
  );
}

/* ------------------------------------------------------------------ */
/* DASHBOARD SCREEN                                                     */
/* ------------------------------------------------------------------ */

function Dashboard(props) {
  const teacher = props.teacher;
  const students = props.students;
  const todayAttendance = props.todayAttendance;

  const searchState = useState("");
  const search = searchState[0], setSearch = searchState[1];
  const dateFilterState = useState(TODAY);
  const dateFilter = dateFilterState[0], setDateFilter = dateFilterState[1];
  const statusFilterState = useState("All");
  const statusFilter = statusFilterState[0], setStatusFilter = statusFilterState[1];
  const attendanceState = useState(todayAttendance);
  const attendance = attendanceState[0], setAttendance = attendanceState[1];
  const liveTodayAttendanceState = useState(todayAttendance);
  const liveTodayAttendance = liveTodayAttendanceState[0], setLiveTodayAttendance = liveTodayAttendanceState[1];
  const lastUpdatedState = useState(null);
  const lastUpdated = lastUpdatedState[0], setLastUpdated = lastUpdatedState[1];
  const connectionState = useState("connecting");
  const connectionStatus = connectionState[0], setConnectionStatus = connectionState[1];
  const dateFilterRef = useRef(dateFilter);
  dateFilterRef.current = dateFilter;
  const refreshRef = useRef(null);

  useEffect(function () {
    let mounted = true;
    let activeController = null;
    let interval = null;

    function stopPolling() {
      if (!mounted) return;
      mounted = false;
      if (interval !== null) clearInterval(interval);
      if (activeController) activeController.abort();
      refreshRef.current = null;
    }

    function refreshAttendance() {
      if (!mounted || teacherAttendancePollInFlight) return;
      teacherAttendancePollInFlight = true;
      const requestedDate = dateFilterRef.current;
      const controller = new AbortController();
      activeController = controller;
      const timeout = setTimeout(function () { controller.abort(); }, 5000);
      const selectedRequest = apiRequest(
        "/api/teacher/attendance?date=" + encodeURIComponent(requestedDate),
        { signal: controller.signal }
      );
      const todayRequest = requestedDate === TODAY
        ? selectedRequest
        : apiRequest("/api/teacher/attendance?date=" + encodeURIComponent(TODAY),
                     { signal: controller.signal });

      return Promise.all([selectedRequest, todayRequest]).then(function (results) {
        if (!mounted) return;
        const selectedRows = uniqueAttendanceRows(results[0]);
        const todayRows = uniqueAttendanceRows(results[1]);
        if (dateFilterRef.current === requestedDate) {
          setAttendance(function (previous) {
            return sameAttendanceRows(previous, selectedRows) ? previous : selectedRows;
          });
        }
        setLiveTodayAttendance(function (previous) {
          return sameAttendanceRows(previous, todayRows) ? previous : todayRows;
        });
        setLastUpdated(new Date());
        setConnectionStatus("connected");
      }).catch(function () {
        if (!mounted) return;
        // Keep the last successful records visible during a temporary outage.
        setConnectionStatus("offline");
      }).finally(function () {
        clearTimeout(timeout);
        if (activeController === controller) activeController = null;
        teacherAttendancePollInFlight = false;
      });
    }

    refreshRef.current = refreshAttendance;
    refreshAttendance();
    interval = setInterval(refreshAttendance, 2000);
    window.addEventListener("beforeunload", stopPolling);
    return function () {
      stopPolling();
      window.removeEventListener("beforeunload", stopPolling);
    };
  }, []);

  useEffect(function () {
    if (refreshRef.current) refreshRef.current();
  }, [dateFilter]);

  function handleRefresh() {
    if (refreshRef.current) refreshRef.current();
  }

  // Build one row per student for the selected date (Absent if no record).
  const rowsForDate = useMemo(function () {
    return students.map(function (student) {
      const rec = attendance.find(function (r) { return r.studentId === student.id; });
      return {
        studentId: student.id,
        date: dateFilter,
        time: rec ? rec.time : "--",
        status: rec ? rec.status : "Absent",
        method: rec ? rec.method : "--",
      };
    });
  }, [students, attendance, dateFilter]);

  const todayRows = useMemo(function () {
    return students.map(function (student) {
      const rec = liveTodayAttendance.find(function (r) { return r.studentId === student.id; });
      return rec ? rec.status : "Absent";
    });
  }, [students, liveTodayAttendance]);

  const totalStudents = students.length;
  const presentToday = todayRows.filter(function (s) { return s === "Present"; }).length;
  const absentToday = todayRows.filter(function (s) { return s === "Absent"; }).length;
  const lateToday = todayRows.filter(function (s) { return s === "Late"; }).length;

  const filteredRows = useMemo(function () {
    return rowsForDate.filter(function (row) {
      const student = studentById(students, row.studentId);
      if (!student) return false;
      if (statusFilter !== "All" && row.status !== statusFilter) return false;
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        const matches = student.name.toLowerCase().indexOf(q) !== -1 ||
                        student.rollNo.toLowerCase().indexOf(q) !== -1;
        if (!matches) return false;
      }
      return true;
    });
  }, [rowsForDate, students, search, statusFilter]);

  function handleExport() {
    downloadCSV(filteredRows, students, "attendance_" + dateFilter + ".csv");
  }

  function handleClearFilters() {
    setSearch("");
    setDateFilter(TODAY);
    setStatusFilter("All");
  }

  return h("div", { className: "screen" },
    h(TopBar, { teacher: teacher, onLogout: props.onLogout }),
    h("div", { className: "app-content" },
      h("div", { className: "greeting" },
        h("h1", null, "Welcome back, " + teacher.name.replace("Dr. ", "").split(" ")[0]),
        h("p", null, "Here's today's attendance overview for your class.")
      ),

      h("div", { className: "stat-cards-grid" },
        h("div", { className: "stat-card" },
          h("div", { className: "stat-card-icon total" }, "\uD83D\uDC65"),
          h("div", null,
            h("div", { className: "num" }, String(totalStudents)),
            h("div", { className: "label" }, "Total students")
          )
        ),
        h("div", { className: "stat-card" },
          h("div", { className: "stat-card-icon present" }, "\u2705"),
          h("div", null,
            h("div", { className: "num" }, String(presentToday)),
            h("div", { className: "label" }, "Present today")
          )
        ),
        h("div", { className: "stat-card" },
          h("div", { className: "stat-card-icon absent" }, "\u274C"),
          h("div", null,
            h("div", { className: "num" }, String(absentToday)),
            h("div", { className: "label" }, "Absent today")
          )
        ),
        h("div", { className: "stat-card" },
          h("div", { className: "stat-card-icon late" }, "\u23F0"),
          h("div", null,
            h("div", { className: "num" }, String(lateToday)),
            h("div", { className: "label" }, "Late today")
          )
        )
      ),

      h("div", { className: "card" },
        h("div", { className: "section-head" },
          h("h3", null, "\uD83D\uDCCB Attendance records"),
        ),

        h("div", { className: "toolbar" },
          h("div", { className: "search-wrap" },
            h("span", { className: "field-icon" }, "\uD83D\uDD0D"),
            h("input", {
              type: "text",
              placeholder: "Search by name or roll number",
              value: search,
              onChange: function (e) { setSearch(e.target.value); }
            })
          ),
          h("input", {
            type: "date",
            value: dateFilter,
            onChange: function (e) { setDateFilter(e.target.value); }
          }),
          h("select", {
            value: statusFilter,
            onChange: function (e) { setStatusFilter(e.target.value); }
          },
            h("option", { value: "All" }, "All statuses"),
            h("option", { value: "Present" }, "Present"),
            h("option", { value: "Absent" }, "Absent"),
            h("option", { value: "Late" }, "Late")
          ),
          h("button", { className: "btn-clear", onClick: handleClearFilters }, "Clear"),
          h("button", { className: "btn-clear", onClick: handleRefresh }, "Refresh"),
          h("button", { className: "btn-export", onClick: handleExport },
            "\u2B07 Export CSV"
          )
        ),

        h("p", { className: "result-count" },
          filteredRows.length + " record" + (filteredRows.length === 1 ? "" : "s") + " for " + dateFilter
        ),
        h("div", {
          role: "status",
          style: { display: "flex", justifyContent: "space-between", gap: "12px", margin: "-8px 0 12px", color: "#68748a", fontSize: "12px" }
        },
          h("span", { style: { color: connectionStatus === "connected" ? "#43866a" : "#9a6b2f" } },
            (connectionStatus === "connected" ? "● Live updates connected" :
              connectionStatus === "offline" ? "● Reconnecting to server" : "● Connecting")
          ),
          h("span", null, lastUpdated ? "Last updated " + lastUpdated.toLocaleTimeString() : "Waiting for first update")
        ),

        h("div", { className: "table-wrap" },
          h("table", { className: "data-table" },
            h("thead", null,
              h("tr", null,
                h("th", null, "Name"),
                h("th", null, "Roll number"),
                h("th", null, "Time"),
                h("th", null, "Method"),
                h("th", null, "Status")
              )
            ),
            h("tbody", null,
              filteredRows.length === 0
                ? h("tr", { className: "empty-row" }, h("td", { colSpan: 5 }, "No matching records."))
                : filteredRows.map(function (row, i) {
                    const student = studentById(students, row.studentId);
                    return h("tr", { key: i, onClick: function () { props.onSelectStudent(row.studentId); } },
                      h("td", { className: "name-cell" }, student.name),
                      h("td", { className: "roll-cell" }, student.rollNo),
                      h("td", null, row.time),
                      h("td", null, row.method),
                      h("td", null, h(StatusPill, { status: row.status }))
                    );
                  })
            )
          )
        ),

        h("div", { className: "mobile-table-list" },
          filteredRows.length === 0
            ? h("div", { className: "mobile-table-row" }, h("span", { className: "m-meta" }, "No matching records."))
            : filteredRows.map(function (row, i) {
                const student = studentById(students, row.studentId);
                return h("div", {
                  className: "mobile-table-row",
                  key: i,
                  onClick: function () { props.onSelectStudent(row.studentId); }
                },
                  h("div", null,
                    h("div", { className: "m-name" }, student.name),
                    h("div", { className: "m-meta" }, student.rollNo + " \u00B7 " + row.time)
                  ),
                  h(StatusPill, { status: row.status })
                );
              })
        )
      )
    )
  );
}

/* ------------------------------------------------------------------ */
/* STUDENT DETAIL SCREEN                                                */
/* ------------------------------------------------------------------ */

function StudentDetail(props) {
  const teacher = props.teacher;
  const student = studentById(props.students, props.studentId);
  const historyState = useState([]);
  const history = historyState[0], setHistory = historyState[1];

  useEffect(function () {
    if (!student) return;
    apiRequest("/api/student/" + encodeURIComponent(student.rollNo) + "/attendance")
      .then(setHistory)
      .catch(function () { setHistory([]); });
  }, [student]);

  const presentCount = history.filter(function (r) { return r.status === "Present"; }).length;
  const lateCount = history.filter(function (r) { return r.status === "Late"; }).length;
  const totalDays = 5; // mock window: today + 4 prior days
  const absentCount = totalDays - presentCount - lateCount;
  const attendancePercent = Math.round(((presentCount + lateCount) / totalDays) * 100);

  const initials = student.name.split(" ").map(function (n) { return n[0]; }).slice(0, 2).join("");

  function handleExport() {
    downloadCSV(history, props.students, student.rollNo + "_attendance.csv");
  }

  return h("div", { className: "screen" },
    h(TopBar, { teacher: teacher, onLogout: props.onLogout }),
    h("div", { className: "app-content" },
      h("button", { className: "back-link", onClick: props.onBack }, "\u2039 Back to dashboard"),

      h("div", { className: "detail-head" },
        h("div", { className: "detail-avatar" }, initials),
        h("div", null,
          h("h1", null, student.name),
          h("p", null, student.rollNo + " \u00B7 " + student.course)
        )
      ),

      h("div", { className: "stat-mini-grid" },
        h("div", { className: "stat-mini" },
          h("div", { className: "num", style: { color: "var(--indigo)" } }, attendancePercent + "%"),
          h("div", { className: "label" }, "Attendance")
        ),
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
          h("h3", null, "\uD83D\uDCC8 Attendance record"),
          h("button", { className: "btn-export", onClick: handleExport }, "\u2B07 Export CSV")
        ),

        h("div", { className: "table-wrap" },
          h("table", { className: "data-table" },
            h("thead", null,
              h("tr", null,
                h("th", null, "Date"),
                h("th", null, "Time"),
                h("th", null, "Method"),
                h("th", null, "Status")
              )
            ),
            h("tbody", null,
              history.length === 0
                ? h("tr", { className: "empty-row" }, h("td", { colSpan: 4 }, "No records yet."))
                : history.map(function (r, i) {
                    return h("tr", { key: i },
                      h("td", { className: "name-cell" }, r.date),
                      h("td", null, r.time),
                      h("td", null, r.method),
                      h("td", null, h(StatusPill, { status: r.status }))
                    );
                  })
            )
          )
        ),

        h("div", { className: "mobile-table-list" },
          history.length === 0
            ? h("div", { className: "mobile-table-row" }, h("span", { className: "m-meta" }, "No records yet."))
            : history.map(function (r, i) {
                return h("div", { className: "mobile-table-row", key: i },
                  h("div", null,
                    h("div", { className: "m-name" }, r.date),
                    h("div", { className: "m-meta" }, r.time + " \u00B7 " + r.method)
                  ),
                  h(StatusPill, { status: r.status })
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
  const screenState = useState("login"); // login | dashboard | detail
  const screen = screenState[0], setScreen = screenState[1];
  const teacherState = useState(null);
  const teacher = teacherState[0], setTeacher = teacherState[1];
  const studentsState = useState([]);
  const students = studentsState[0], setStudents = studentsState[1];
  const todayAttendanceState = useState([]);
  const todayAttendance = todayAttendanceState[0], setTodayAttendance = todayAttendanceState[1];
  const selectedStudentState = useState(null);
  const selectedStudentId = selectedStudentState[0], setSelectedStudentId = selectedStudentState[1];

  function handleLogin(data) {
    return apiRequest("/api/teacher/students").then(function (studentRows) {
      setTeacher(data);
      setStudents(studentRows);
      // Dashboard's mount-time poll fetches today's attendance once and then
      // continues on the existing two-second interval.
      setTodayAttendance([]);
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
    if (role !== "teacher") {
      window.location.replace("../index.html");
      return;
    }
    try {
      const user = JSON.parse(sessionStorage.getItem("smartAttendanceUser") || "{}");
      if (!user.username) throw new Error("Missing teacher session identity.");
      handleLogin(user).catch(function () {
        clearAuthSession();
        setTeacher(null);
        setScreen("login");
      });
    } catch (_error) {
      clearAuthSession();
      setScreen("login");
    }
  }, []);

  function handleSelectStudent(id) {
    setSelectedStudentId(id);
    setScreen("detail");
  }

  if (screen === "login") return h(LoginScreen, { onLogin: handleLogin });

  if (screen === "dashboard") return h(Dashboard, {
    teacher: teacher,
    students: students,
    todayAttendance: todayAttendance,
    onSelectStudent: handleSelectStudent,
    onLogout: handleLogout
  });

  if (screen === "detail") return h(StudentDetail, {
    teacher: teacher,
    studentId: selectedStudentId,
    students: students,
    onBack: function () { setScreen("dashboard"); },
    onLogout: handleLogout
  });

  return null;
}

const rootEl = document.getElementById("root");
const root = ReactDOM.createRoot(rootEl);
root.render(h(App));
