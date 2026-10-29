# Score, verify and CI-layer a set of cases in parallel (one PowerShell job per case).
#   .\run_cases.ps1 2024-03-25 2024-09-12 ...
$py = "C:\Users\hunte\AppData\Local\Programs\Python\Python313\python.exe"
$proto = $PSScriptRoot
$jobs = foreach ($d in $args) {
    Start-Job -ArgumentList $py, $proto, $d -ScriptBlock {
        param($py, $proto, $d)
        Set-Location $proto
        $log = "..\cases\$d\logs"
        New-Item -ItemType Directory -Force $log | Out-Null
        & $py run_case.py $d --leads 8 7 6 5 4 3 2 1 *> "$log\run_case.log"
        & $py verify_case.py $d *> "$log\verify.log"
        & $py ci.py $d --leads 8 7 6 5 4 3 2 1 *> "$log\ci.log"
        "$d done"
    }
}
$jobs | Wait-Job | Receive-Job
