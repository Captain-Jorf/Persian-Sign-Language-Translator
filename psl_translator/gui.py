from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Button, Entry, Frame, Label, LabelFrame, StringVar, Text, Tk, filedialog, messagebox, ttk

from psl_translator.dataset import write_jsonl
from psl_translator.dictionary import sentence_to_sign_tokens
from psl_translator.islr101 import facts as islr101_facts
from psl_translator.toy_data import make_toy_samples
from psl_translator.trainer import train_evaluate_report


class PSLTranslatorApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Persian Sign Language Translator")
        self.root.geometry("920x680")
        self.root.minsize(760, 560)

        self.model_path = StringVar(value="models/prototype_model.json")
        self.report_dir = StringVar(value="reports/demo")
        self.status = StringVar(value="Ready. Use Text-to-Sign or train demo model.")
        self.camera_running = False

        notebook = ttk.Notebook(root)
        notebook.pack(fill=BOTH, expand=True, padx=12, pady=12)

        self._build_text_tab(notebook)
        self._build_training_tab(notebook)
        self._build_camera_tab(notebook)
        self._build_about_tab(notebook)

        footer = Frame(root)
        footer.pack(fill=X, padx=12, pady=(0, 10))
        Label(footer, textvariable=self.status, anchor="w").pack(side=LEFT, fill=X, expand=True)

    def _build_text_tab(self, notebook: ttk.Notebook) -> None:
        tab = Frame(notebook)
        notebook.add(tab, text="Text to Sign")

        Label(tab, text="Persian sentence:").pack(anchor="w", padx=14, pady=(14, 4))
        self.text_input = Text(tab, height=5, wrap="word")
        self.text_input.pack(fill=X, padx=14)
        self.text_input.insert("1.0", "سلام لطفا آب")

        Button(tab, text="Convert to PSL gloss", command=self._convert_text).pack(anchor="w", padx=14, pady=10)

        Label(tab, text="Output:").pack(anchor="w", padx=14, pady=(4, 4))
        self.text_output = Text(tab, height=18, wrap="none")
        self.text_output.pack(fill=BOTH, expand=True, padx=14, pady=(0, 14))

    def _build_training_tab(self, notebook: ttk.Notebook) -> None:
        tab = Frame(notebook)
        notebook.add(tab, text="Train / Report")

        box = LabelFrame(tab, text="Demo training")
        box.pack(fill=X, padx=14, pady=14)
        Label(
            box,
            text="This trains the bundled toy pipeline and generates HTML/SVG reports. It is a smoke test, not real ISLR101 accuracy.",
            anchor="w",
            justify=LEFT,
        ).pack(fill=X, padx=10, pady=(8, 4))
        Button(box, text="Train demo model + create report", command=self._train_demo_async).pack(anchor="w", padx=10, pady=(4, 10))

        paths = LabelFrame(tab, text="Paths")
        paths.pack(fill=X, padx=14, pady=(0, 14))
        self._path_row(paths, "Model", self.model_path, self._choose_model)
        self._path_row(paths, "Report", self.report_dir, self._choose_report_dir)

        actions = Frame(tab)
        actions.pack(fill=X, padx=14)
        Button(actions, text="Open report", command=self._open_report).pack(side=LEFT)
        Button(actions, text="Open report folder", command=self._open_report_folder).pack(side=LEFT, padx=8)

        Label(tab, text="Log:").pack(anchor="w", padx=14, pady=(14, 4))
        self.train_log = Text(tab, height=14, wrap="word")
        self.train_log.pack(fill=BOTH, expand=True, padx=14, pady=(0, 14))

    def _build_camera_tab(self, notebook: ttk.Notebook) -> None:
        tab = Frame(notebook)
        notebook.add(tab, text="Camera")

        Label(
            tab,
            text="Webcam mode needs OpenCV + MediaPipe inside the build. Use a trained model first. Press q in the camera window to stop.",
            justify=LEFT,
            wraplength=820,
        ).pack(anchor="w", padx=14, pady=14)

        model_box = LabelFrame(tab, text="Model")
        model_box.pack(fill=X, padx=14, pady=(0, 14))
        self._path_row(model_box, "Model", self.model_path, self._choose_model)

        Button(tab, text="Start camera translation", command=self._start_camera_async).pack(anchor="w", padx=14)

    def _build_about_tab(self, notebook: ttk.Notebook) -> None:
        tab = Frame(notebook)
        notebook.add(tab, text="ISLR101 / About")

        Label(
            tab,
            text="ISLR101 dataset facts from arXiv:2503.12451. The actual archive must be requested from the lab.",
            justify=LEFT,
            wraplength=840,
        ).pack(anchor="w", padx=14, pady=14)

        output = Text(tab, height=22, wrap="word")
        output.pack(fill=BOTH, expand=True, padx=14, pady=(0, 14))
        output.insert("1.0", json.dumps(islr101_facts(), ensure_ascii=False, indent=2))
        output.configure(state="disabled")

    def _path_row(self, parent: Frame, label: str, variable: StringVar, command) -> None:
        row = Frame(parent)
        row.pack(fill=X, padx=10, pady=8)
        Label(row, text=f"{label}:", width=9, anchor="w").pack(side=LEFT)
        Entry(row, textvariable=variable).pack(side=LEFT, fill=X, expand=True, padx=6)
        Button(row, text="Browse", command=command).pack(side=RIGHT)

    def _convert_text(self) -> None:
        text = self.text_input.get("1.0", END).strip()
        tokens = sentence_to_sign_tokens(text)
        self.text_output.delete("1.0", END)
        self.text_output.insert("1.0", json.dumps([asdict(token) for token in tokens], ensure_ascii=False, indent=2))
        self.status.set(f"Converted {len(tokens)} token(s).")

    def _train_demo_async(self) -> None:
        self._run_background(self._train_demo, "Training demo model...")

    def _train_demo(self) -> None:
        data_dir = Path("data/demo")
        model_path = Path(self.model_path.get())
        report_dir = Path(self.report_dir.get())

        data_dir.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        train = make_toy_samples(per_label=8, frames=20)
        eval_ = make_toy_samples(per_label=3, frames=16)

        write_jsonl(train, data_dir / "toy_train.jsonl")
        write_jsonl(eval_, data_dir / "toy_eval.jsonl")

        report = train_evaluate_report(
            train_dataset=data_dir / "toy_train.jsonl",
            eval_dataset=data_dir / "toy_eval.jsonl",
            model_out=model_path,
            report_dir=report_dir,
        )

        self._append_log(json.dumps(report, ensure_ascii=False, indent=2))
        self.status.set(f"Demo trained. Accuracy: {report['accuracy']:.2%}")

    def _start_camera_async(self) -> None:
        if self.camera_running:
            messagebox.showinfo("Camera already running", "Camera mode is already running. Press q in the camera window to stop it.")
            return

        model = Path(self.model_path.get())
        if not model.exists():
            messagebox.showerror("Model missing", f"Model not found:\n{model}\n\nTrain demo model first or choose a model file.")
            return

        self.camera_running = True
        self._run_background(lambda: self._start_camera(model), "Starting camera...")

    def _find_project_root(self) -> Path:
        candidates: list[Path] = []

        try:
            candidates.append(Path.cwd())
        except Exception:
            pass

        if getattr(sys, "frozen", False):
            exe = Path(sys.executable).resolve()
            candidates.extend([exe.parent, exe.parent.parent, exe.parent.parent.parent])
        else:
            here = Path(__file__).resolve()
            candidates.extend([here.parent, here.parent.parent, here.parent.parent.parent])

        for candidate in candidates:
            if (candidate / "psl_translator").exists() and (candidate / "models").exists():
                return candidate

        return Path.cwd()

    def _find_runtime_python(self, project_root: Path) -> Path:
        candidates = [
            project_root / ".venv" / "Scripts" / "python.exe",
            project_root / ".venv311" / "Scripts" / "python.exe",
            project_root / ".packvenv" / "Scripts" / "python.exe",
        ]

        for python_path in candidates:
            if not python_path.exists():
                continue

            check = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    "import cv2, mediapipe as mp; import sys; sys.exit(0 if hasattr(cv2, 'VideoCapture') and hasattr(mp, 'solutions') else 1)",
                ],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if check.returncode == 0:
                return python_path

        found = shutil.which("python")
        if found:
            return Path(found)

        raise RuntimeError("No Python runtime with OpenCV and MediaPipe was found.")

    def _start_camera(self, model: Path) -> None:
        # Do not import cv2/mediapipe inside the PyInstaller exe.
        # OpenCV packaging is the problem. We run camera with the project venv instead.
        project_root = self._find_project_root()
        python_path = self._find_runtime_python(project_root)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        try:
            subprocess.run(
                [
                    str(python_path),
                    "-m",
                    "psl_translator",
                    "camera",
                    "--model",
                    str(model.resolve()),
                ],
                cwd=project_root,
                env=env,
                check=False,
            )
        finally:
            self.camera_running = False
            self.status.set("Camera stopped.")

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(title="Choose model", filetypes=[("JSON model", "*.json"), ("All files", "*.*")])
        if path:
            self.model_path.set(path)

    def _choose_report_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose report directory")
        if path:
            self.report_dir.set(path)

    def _open_report(self) -> None:
        report = Path(self.report_dir.get()) / "index.html"
        if not report.exists():
            messagebox.showwarning("Report missing", f"Report not found:\n{report}\n\nTrain demo model first.")
            return
        webbrowser.open(report.resolve().as_uri())

    def _open_report_folder(self) -> None:
        folder = Path(self.report_dir.get())
        folder.mkdir(parents=True, exist_ok=True)
        webbrowser.open(folder.resolve().as_uri())

    def _run_background(self, target, busy_message: str) -> None:
        self.status.set(busy_message)

        def runner() -> None:
            try:
                target()
            except Exception as exc:
                self._append_log(f"ERROR: {exc}")
                self.status.set("Failed. Check log.")
                messagebox.showerror("Persian Sign Language Translator", str(exc))

        threading.Thread(target=runner, daemon=True).start()

    def _append_log(self, message: str) -> None:
        def update() -> None:
            self.train_log.insert(END, message + "\n")
            self.train_log.see(END)

        self.root.after(0, update)


def main() -> int:
    root = Tk()
    PSLTranslatorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
