async function sanitizeText() {
    let inputText = document.getElementById("inputText").value;
    let response = await fetch("/sanitize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
    });
    let result = await response.json();
    document.getElementById("inputText").value = result.sanitized_text;
}

function copyToClipboard() {
    let textArea = document.getElementById("inputText");

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(textArea.value)
            .then(() => showSnackbar("Text copied to clipboard"))
            .catch(() => showSnackbar("Failed to copy text"));
    } else {
        textArea.select();
        document.execCommand("copy"); // Deprecated fallback
        showSnackbar("Text copied to clipboard");
    }
}

function clearText() {
    let textArea = document.getElementById("inputText");
    textArea.value = "";
}

function showSnackbar(message) {
    let snackbar = document.getElementById("snackbar");
    snackbar.textContent = message;
    snackbar.classList.add("show");

    setTimeout(() => {
        snackbar.classList.remove("show");
    }, 3000); // Hide after 3 seconds
}