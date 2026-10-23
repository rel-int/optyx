import modal, subprocess, os
app = modal.App("paper-build")
image = (modal.Image.debian_slim()
         .apt_install("texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra",
                      "texlive-publishers", "texlive-science", "texlive-pictures", "texlive-fonts-recommended",
                      "texlive-bibtex-extra", "ghostscript")
         .add_local_dir("paper", remote_path="/root/paper"))

@app.function(image=image, timeout=1200)
def build():
    os.chdir("/root/paper")
    log = []
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.append(f"$ {' '.join(cmd)} -> {r.returncode}\n" + r.stdout[-3000:] + r.stderr[-1000:])
    pdf = open("main.pdf", "rb").read() if os.path.exists("main.pdf") else b""
    return pdf, "\n".join(log), open("main.log").read() if os.path.exists("main.log") else ""

@app.local_entrypoint()
def main():
    pdf, log, full = build.remote()
    open("main.pdf", "wb").write(pdf)
    open("build.log", "w").write(full)
    print(log[-6000:])
    print("pdf bytes", len(pdf))
