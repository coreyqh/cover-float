import traceback
from pathlib import Path
from typing import Callable, TextIO

import cover_float.common.log as log
from cover_float.scripts.parse_testvectors import auto_parse

GLOBAL_MODELS: dict[str, Callable[[Path], None]] = {}


def register_model(model_name: str) -> Callable[[Callable[[TextIO, TextIO], None]], Callable[[Path], None]]:
    def inner(fn: Callable[[TextIO, TextIO], None]) -> Callable[[Path], None]:
        def wrapper(output_dir: Path, post_process: bool = True) -> None:
            tv_path = output_dir / "testvectors" / f"{model_name}_tv.txt"
            cv_path = output_dir / "covervectors" / f"{model_name}_cv.txt"

            log.set_prefix(f"{model_name} Test Generation: ")

            with tv_path.open("w") as test_f, cv_path.open("w") as cover_f:
                try:
                    fn(test_f, cover_f)
                except Exception:
                    print(f"\033[KFatal Error in {model_name}")
                    traceback.print_exc()

            log.set_prefix(f"{model_name} Post Processing: ")

            if post_process:
                auto_parse(model_name, str(output_dir))

        GLOBAL_MODELS[model_name] = wrapper
        return wrapper

    return inner
