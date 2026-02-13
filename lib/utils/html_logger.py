import os
import matplotlib.pyplot as plt
from pathlib import Path

class HTMLFigureLogger:
    def __init__(self, log_dir='html_log', html_name='index.html', comment="logged_figures"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        html_base      = Path(html_name).stem
        self.images_dir = self.log_dir / f"{html_base}_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.html_path = self.log_dir / html_name
        self.comment   = comment
        self._init_html()
        self.entries = []

    def _init_html(self):
        with open(self.html_path, "w") as f:
            f.write("<html><head><title>Figure Log</title></head><body>\n")
            f.write(f"<h1>{self.comment}</h1>\n")
            f.write("<style>img {width: 200px; margin: 5px;} "
                    ".row {display: flex; flex-wrap: wrap;}</style>\n")

    def add_figure(self, tag, fig, global_step):
        img_filename = f"{tag}_{global_step}.png".replace("/", "_")
        img_path     = self.images_dir / img_filename
        fig.savefig(img_path, dpi=300,bbox_inches='tight')
        plt.close(fig)
        self.entries.append((img_path, img_filename, tag, global_step))

    def finalize(self):
        with open(self.html_path, 'a') as f:
            for i, (img_path, fname, tag, step) in enumerate(self.entries):
                if i % 6 == 0:
                    if i > 0:    # close previous row
                        f.write("</div>\n")
                    f.write("<div class='row'>\n")

                # ---- key line: make src *relative* to html file ----
                rel_src = os.path.relpath(img_path, start=self.html_path.parent)
                f.write(
                    f"<div><img src='{rel_src}' alt='{tag} step {step}'><br>"
                    f"{tag} (Step {step})</div>\n"
                )

            f.write("</div>\n</body></html>\n")

