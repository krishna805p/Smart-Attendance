(function () {
  "use strict";

  const API_BASE = window.SMART_ATTENDANCE_CONFIG.apiBaseUrl;
  const FRONTENDS = {
    student: "./Student_Frontend/index.html",
    teacher: "./Teacher_Frontend/teacher.html"
  };
  let selectedRole = "student";

  const roleButtons = Array.from(document.querySelectorAll(".role-option"));
  const form = document.getElementById("login-form");
  const identity = document.getElementById("identity");
  const identityLabel = document.getElementById("identity-label");
  const password = document.getElementById("password");
  const errorBox = document.getElementById("login-error");
  const submitButton = document.getElementById("submit-button");

  function clearSession() {
    ["smartAttendanceToken", "smartAttendanceRole", "smartAttendanceUser"].forEach(function (key) {
      sessionStorage.removeItem(key);
    });
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  roleButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      selectedRole = button.dataset.role;
      roleButtons.forEach(function (other) {
        const active = other === button;
        other.classList.toggle("active", active);
        other.setAttribute("aria-pressed", active ? "true" : "false");
      });
      identityLabel.textContent = selectedRole === "student" ? "Roll number" : "Username";
      identity.autocomplete = "username";
      identity.placeholder = selectedRole === "student" ? "Your roll number" : "Your faculty username";
      submitButton.textContent = "Sign in as " + (selectedRole === "student" ? "Student" : "Teacher");
      errorBox.hidden = true;
    });
  });

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    errorBox.hidden = true;
    submitButton.disabled = true;
    submitButton.textContent = "Signing in…";
    clearSession();

    const student = selectedRole === "student";
    const body = student
      ? { roll_no: identity.value.trim(), password: password.value }
      : { username: identity.value.trim(), password: password.value };
    const endpoint = student ? "/api/student/login" : "/api/teacher/login";

    try {
      const response = await fetch(API_BASE + endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Unable to sign in.");
      if (!result.access_token || result.token_type !== "bearer") {
        throw new Error("The server did not provide a usable sign-in session.");
      }

      const user = Object.assign({}, result);
      delete user.access_token;
      delete user.token_type;
      sessionStorage.setItem("smartAttendanceToken", result.access_token);
      sessionStorage.setItem("smartAttendanceRole", selectedRole);
      sessionStorage.setItem("smartAttendanceUser", JSON.stringify(user));
      window.location.assign(FRONTENDS[selectedRole]);
    } catch (error) {
      clearSession();
      showError(error.message || "Unable to sign in. Check the backend connection and try again.");
      submitButton.disabled = false;
      submitButton.textContent = "Sign in as " + (selectedRole === "student" ? "Student" : "Teacher");
    }
  });
})();
