Start-Process powershell -ArgumentList '-NoExit', '-File', '.\\run_server.ps1'
Start-Sleep -Seconds 1
Start-Process powershell -ArgumentList '-NoExit', '-File', '.\\run_client1.ps1'
Start-Sleep -Seconds 1
Start-Process powershell -ArgumentList '-NoExit', '-File', '.\\run_client2.ps1'