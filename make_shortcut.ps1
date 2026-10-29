# Creates "Drums2Click.lnk" next to this script: opens the GUI without a console window, with the app icon.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path $here "Drums2Click.lnk"))
$link.TargetPath = Join-Path $here ".venv\Scripts\pythonw.exe"
$link.Arguments = '"' + (Join-Path $here "gui.py") + '"'
$link.WorkingDirectory = $here
$link.IconLocation = (Join-Path $here "assets\icon.ico") + ",0"
$link.Description = "Drums -> Click: dallo stem di batteria a un click che segue il batterista"
$link.Save()
Write-Host "Created Drums2Click.lnk (you can copy it to the desktop or pin it to Start)"
