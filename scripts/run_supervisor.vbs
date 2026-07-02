' Launches the traffic daylight-supervisor with NO console window, for the
' TrafficSupervisor scheduled task. WshShell.Run(..., 0, False) runs hidden, so
' nothing pops up on the desktop at logon. Self-locates the repo root (this file
' lives in <root>\scripts), redirecting all output to data\logs\supervise.log.
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = root
cmd = "cmd /c ""set HF_HUB_DISABLE_SYMLINKS_WARNING=1&& .venv\Scripts\python.exe -m traffic_logger.main supervise --config config\config.run.local.yaml >> data\logs\supervise.log 2>&1"""
sh.Run cmd, 0, False
