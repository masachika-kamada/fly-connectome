"""Build the pages into dist/, and assemble the publishable site into site/.

Every page carries its data (and shared code) inline, so each works as a plain
file, on the local server, and on any static host, with no fetch and no CORS.

  web/trainer.template.html + data/odor_kc.json                 -> dist/trainer.html
  web/errand.template.html  + data/errand.json + errand-core.js -> dist/errand.html

site/ is what GitHub Pages serves (.github/workflows/pages.yml builds it on
every push to main). The names are flat and every link is relative, because
Pages serves the site under /fly-connectome/, not at the root:

  site/index.html      the errand
  site/brain.html      inside the mushroom body (the trainer)
  site/explainer.html  the explainer animation
  site/figures.html    the result figures, with site/figures/*.svg
"""
import io
import os
import shutil

from flymb.paths import DATA, DIST, FIGURES, ODOR_KC, ROOT, WEB, ensure

DATA_MARK = "/*__DATA__*/ null"
CORE_MARK = "/*__CORE__*/"
SITE = os.path.join(ROOT, "site")

PAGES = [
    ("trainer.template.html", ODOR_KC, None, "trainer.html"),
    ("errand.template.html", os.path.join(DATA, "errand.json"),
     os.path.join(WEB, "errand-core.js"), "errand.html"),
]

SITE_FILES = {
    "index.html": os.path.join(DIST, "errand.html"),
    "brain.html": os.path.join(DIST, "trainer.html"),
    "explainer.html": os.path.join(WEB, "artifact.html"),
    "figures.html": os.path.join(WEB, "figures.html"),
}


def read(path):
    return io.open(path, encoding="utf-8").read()


def build_pages():
    ensure(DIST)
    for template, data, core, out in PAGES:
        html = read(os.path.join(WEB, template))
        if DATA_MARK not in html:
            raise SystemExit(f"{template}: data placeholder not found")
        html = html.replace(DATA_MARK, read(data))
        if core:
            if CORE_MARK not in html:
                raise SystemExit(f"{template}: code placeholder not found")
            # A literal "</script>" inside the code would end the inline tag early.
            code = read(core)
            if "</script" in code:
                raise SystemExit(f"{core}: contains </script>, cannot be inlined")
            html = html.replace(CORE_MARK, code)
        path = os.path.join(DIST, out)
        io.open(path, "w", encoding="utf-8").write(html)
        print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")


def assemble_site():
    shutil.rmtree(SITE, ignore_errors=True)
    ensure(SITE, os.path.join(SITE, "figures"))
    for name, src in SITE_FILES.items():
        shutil.copyfile(src, os.path.join(SITE, name))
    for f in sorted(os.listdir(FIGURES)):
        if f.endswith(".svg"):
            shutil.copyfile(os.path.join(FIGURES, f), os.path.join(SITE, "figures", f))
    # Serve the files as they are; no Jekyll processing on Pages.
    io.open(os.path.join(SITE, ".nojekyll"), "w").close()
    print(f"assembled {SITE}  ({', '.join(SITE_FILES)} + figures/)")


if __name__ == "__main__":
    build_pages()
    assemble_site()
