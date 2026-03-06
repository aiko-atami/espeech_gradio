# Keep CSS scoped to our own Gradio element ids. Internal component DOM
# changes between Gradio releases and breaks spacing/layout hacks.
APP_CSS = """
.padded-text {
    padding: 4px 8px;
    background: transparent;
    border: 0px;
    box-shadow: none;
}

#ref-inputs-row {
    align-items: stretch;
}

.no-border-accordion {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.no-border-accordion > button.label-wrap {
    border: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.no-border-accordion > div.wrap,
.no-border-accordion > div.contain {
    border: none !important;
    background: transparent !important;
}

/* Set overall app width and center */
.gradio-container {
    width: 100% !important;
    max-width: 1280px !important;
    margin: auto !important;
}
"""
