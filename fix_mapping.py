from pathlib import Path

path = Path("investigation_cases.py")
source = path.read_text(encoding="utf-8")

start_marker = '    erp = erp.reset_index(drop=True)'
end_marker = '    cases["real_sensor_erp_link_verified"] = False'

start = source.find(start_marker)

if start == -1:
    raise SystemExit("Old block start not found. No changes made.")

end = source.find(end_marker, start)

if end == -1:
    raise SystemExit("Old block end not found. No changes made.")

end += len(end_marker)
old_block = source[start:end]

if "sensor_indices" not in old_block or "assigned_sensor" not in old_block:
    raise SystemExit("Safety check failed. No changes made.")

backup = Path("investigation_cases.py.backup")
backup.write_text(source, encoding="utf-8")

path.write_text(source[:start] + source[end:], encoding="utf-8")

print("Old round-robin block removed successfully.")
print("Backup: investigation_cases.py.backup")