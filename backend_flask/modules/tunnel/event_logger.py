import csv
from datetime import datetime
from pathlib import Path


CSV_COLUMNS = [
    "event_id",
    "event_date",
    "event_time",
    "event_datetime",
    "cctv_name",
    "event_type",
    "event_status",
    "operator_action",
    "frame_id",
    "traffic_state",
    "avg_speed",
    "vehicle_count",
    "lane_count",
    "reason",
    "capture_path",
]


class TunnelEventLogger:
    def __init__(self, runtime_root=None):
        tunnel_dir = Path(__file__).resolve().parent
        self.runtime_root = Path(runtime_root) if runtime_root else tunnel_dir / "runtime_data"
        self.log_dir = self.runtime_root / "logs"
        self.csv_path = self.log_dir / "tunnel_event_log.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv()

    def _ensure_csv(self):
        if self.csv_path.exists():
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

    def _read_rows(self):
        self._ensure_csv()
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _write_rows(self, rows):
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})

    def append_suspect_event(self, event):
        rows = self._read_rows()
        if any(row.get("event_id") == event.get("event_id") for row in rows):
            return

        row = {key: event.get(key, "") for key in CSV_COLUMNS}
        row["event_type"] = row.get("event_type") or "ACCIDENT"
        row["event_status"] = row.get("event_status") or "SUSPECT"
        rows.append(row)
        self._write_rows(rows)

    def resolve_event(self, event_id, event_status, operator_action):
        rows = self._read_rows()
        resolved = None
        now = datetime.now()

        for row in rows:
            if row.get("event_id") != event_id:
                continue

            row["event_status"] = event_status
            row["operator_action"] = operator_action
            resolved = dict(row)
            break

        if resolved is None:
            resolved = {
                "event_id": event_id,
                "event_date": now.strftime("%Y-%m-%d"),
                "event_time": now.strftime("%H:%M:%S"),
                "event_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "ACCIDENT",
                "event_status": event_status,
                "operator_action": operator_action,
            }
            rows.append(resolved)

        self._write_rows(rows)
        return resolved

    def get_stats(self, date_text=None):
        rows = self._read_rows()
        target_date = date_text or datetime.now().strftime("%Y-%m-%d")
        day_rows = [row for row in rows if row.get("event_date") == target_date]

        total_suspect = len(day_rows)
        confirmed = sum(1 for row in day_rows if row.get("event_status") == "CONFIRMED")
        false_alarm = sum(1 for row in day_rows if row.get("event_status") == "FALSE_ALARM")
        processed = confirmed + false_alarm

        recent_events = [
            row for row in day_rows
            if row.get("event_status") in ("CONFIRMED", "FALSE_ALARM")
        ]
        recent_events.sort(key=lambda row: row.get("event_datetime", ""), reverse=True)

        return {
            "date": target_date,
            "total_suspect": total_suspect,
            "confirmed": confirmed,
            "false_alarm": false_alarm,
            "confirm_rate": round((confirmed / processed) * 100, 1) if processed else 0.0,
            "false_alarm_rate": round((false_alarm / processed) * 100, 1) if processed else 0.0,
            "recent_events": recent_events[:5],
        }
