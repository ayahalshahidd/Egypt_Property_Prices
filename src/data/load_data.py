from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

DEFAULT_SPARK_MASTER = "local-cluster[2,2,4096]"

FALLBACK_SPARK_MASTER = "local[2]"


def configure_local_spark(java_home: str | None = None, temp_dir: str | Path | None = None) -> None:
    """Set local Spark defaults that make PySpark easier to run on Windows."""
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")

    if temp_dir is None:
        temp_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Temp" / "egypt_property_spark"
    temp_path = Path(temp_dir).resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_path)
    os.environ["TMP"] = str(temp_path)
    os.environ["TMPDIR"] = str(temp_path)
    tempfile.tempdir = str(temp_path)

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark(
    app_name: str = "EgyptPropertyPrices",
    master: str | None = None,
    allow_fallback: bool = True,
) -> SparkSession:
    """Create a Spark session.

    The default master uses Spark's pseudo-distributed local-cluster mode:
    one machine, multiple worker processes. If that mode fails on a local
    Windows setup, callers can allow fallback to regular local multicore mode.
    """
    selected_master = master or os.environ.get("SPARK_MASTER") or DEFAULT_SPARK_MASTER
    try:
        return _build_spark(app_name, selected_master)
    except Exception:
        if not allow_fallback or selected_master == FALLBACK_SPARK_MASTER:
            raise
        print(
            f"Spark failed to start with master={selected_master!r}; "
            f"falling back to {FALLBACK_SPARK_MASTER!r}."
        )
        return _build_spark(app_name, FALLBACK_SPARK_MASTER)


def _build_spark(app_name: str, master: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.master", master)
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "2g"))
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "4"))
        .config("spark.default.parallelism", os.environ.get("SPARK_DEFAULT_PARALLELISM", "4"))
        .getOrCreate()
    )


def load_raw_data(spark: SparkSession, path: str | Path) -> DataFrame:
    return spark.read.csv(str(path), header=True, inferSchema=True)


def write_csv(df: DataFrame, output_dir: str | Path, coalesce: bool = True) -> None:
    """Write a Spark DataFrame as CSV, falling back to pandas only on local FS failures."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    writer_df = df.coalesce(1) if coalesce else df
    try:
        if platform.system() == "Windows" and not os.environ.get("HADOOP_HOME"):
            raise RuntimeError("HADOOP_HOME is not set; using pandas CSV export on Windows.")
        writer_df.write.mode("overwrite").option("header", True).csv(str(output_dir))
    except Exception as exc:
        print(f"Spark CSV write skipped/failed for {output_dir}: {exc}")
        print("Falling back to pandas CSV export for this local artifact.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "part-00000.csv"
        df.toPandas().to_csv(output_file, index=False, encoding="utf-8-sig")
