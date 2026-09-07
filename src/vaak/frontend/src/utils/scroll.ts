export function scrollToListen() {
  const introEl = document.querySelector(".listen-intro") || document.getElementById("listen");
  if (introEl) {
    const top = introEl.getBoundingClientRect().top + window.scrollY - 28;
    window.scrollTo({
      top: Math.max(0, Math.round(top)),
      behavior: "smooth"
    });
  }
}
