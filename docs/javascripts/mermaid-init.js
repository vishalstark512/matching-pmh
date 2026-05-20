/* Render Mermaid diagrams on Material for MkDocs page loads. */
document$.subscribe(function () {
  if (typeof mermaid === "undefined") {
    return;
  }
  mermaid.initialize({
    startOnLoad: false,
    theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default",
  });
  mermaid.run({ querySelector: ".mermaid" });
});
