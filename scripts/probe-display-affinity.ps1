<#
.SYNOPSIS
  Print the display affinity and extended styles of every top-level window owned
  by a running gimme-a-chance process.

.DESCRIPTION
  `contentProtected: true` in tauri.conf.json makes tao call
  SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE). That call can fail
  silently (it needs Windows 10 2004 / build 19041+), leaving the window fully
  capturable. This reads the affinity back from the OS, so "is the overlay
  really protected?" stops being a guess.

  Affinity values (winuser.h):
    0x00  WDA_NONE                — captured normally
    0x01  WDA_MONITOR             — capture renders the window BLACK
    0x11  WDA_EXCLUDEFROMCAPTURE  — window omitted from capture entirely

  Extended styles reported because they matter for the stealth overlay:
    WS_EX_NOACTIVATE (0x08000000) — window never takes foreground focus
    WS_EX_TOOLWINDOW (0x00000080) — window absent from taskbar and Alt+Tab

.EXAMPLE
  .\scripts\probe-display-affinity.ps1
  .\scripts\probe-display-affinity.ps1 -ProcessName chrome
#>
param(
    [string]$ProcessName = 'gimme-a-chance'
)

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class WinProbe {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll")] public static extern bool GetWindowDisplayAffinity(IntPtr hWnd, out uint affinity);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr hWnd, StringBuilder s, int max);
    [DllImport("user32.dll", EntryPoint = "GetWindowLongPtrW")] public static extern IntPtr GetWindowLongPtr(IntPtr hWnd, int idx);

    public static List<IntPtr> TopLevelWindowsOf(uint targetPid) {
        var found = new List<IntPtr>();
        EnumWindows((h, l) => {
            uint pid;
            GetWindowThreadProcessId(h, out pid);
            if (pid == targetPid) found.Add(h);
            return true;
        }, IntPtr.Zero);
        return found;
    }

    public static string TitleOf(IntPtr h) {
        var sb = new StringBuilder(512);
        GetWindowTextW(h, sb, sb.Capacity);
        return sb.ToString();
    }
}
"@

function Format-Affinity([uint32]$a) {
    switch ($a) {
        0x00 { 'WDA_NONE               (captured normally — NOT protected)' }
        0x01 { 'WDA_MONITOR            (capture renders BLACK)' }
        0x11 { 'WDA_EXCLUDEFROMCAPTURE (omitted from capture)' }
        default { "unknown (0x{0:X})" -f $a }
    }
}

$procs = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
if (-not $procs) {
    Write-Warning "No process named '$ProcessName' is running. Start it first (.\scripts\dev.ps1)."
    exit 1
}

$GWL_EXSTYLE = -20
$WS_EX_NOACTIVATE = [uint64]0x08000000
$WS_EX_TOOLWINDOW = [uint64]0x00000080

foreach ($p in $procs) {
    foreach ($hwnd in [WinProbe]::TopLevelWindowsOf([uint32]$p.Id)) {
        $title = [WinProbe]::TitleOf($hwnd)
        # Tauri creates message-only / helper windows with no title; skip the noise.
        if ([string]::IsNullOrWhiteSpace($title)) { continue }

        $affinity = 0
        $ok = [WinProbe]::GetWindowDisplayAffinity($hwnd, [ref]$affinity)
        $ex = [uint64][WinProbe]::GetWindowLongPtr($hwnd, $GWL_EXSTYLE)

        $flags = @()
        if ($ex -band $WS_EX_NOACTIVATE) { $flags += 'NOACTIVATE' }
        if ($ex -band $WS_EX_TOOLWINDOW) { $flags += 'TOOLWINDOW' }
        if ([WinProbe]::IsWindowVisible($hwnd)) { $flags += 'VISIBLE' }
        if ([WinProbe]::IsIconic($hwnd)) { $flags += 'MINIMIZED' }

        Write-Output ''
        Write-Output ("hwnd 0x{0:X}  pid {1}" -f [int64]$hwnd, $p.Id)
        Write-Output ("  title    : {0}" -f $title)
        if ($ok) {
            Write-Output ("  affinity : 0x{0:X2}  {1}" -f $affinity, (Format-Affinity $affinity))
        } else {
            Write-Output ("  affinity : <GetWindowDisplayAffinity failed: {0}>" -f [ComponentModel.Win32Exception][Runtime.InteropServices.Marshal]::GetLastWin32Error())
        }
        Write-Output ("  exstyle  : 0x{0:X8}  [{1}]" -f $ex, ($flags -join ' '))
    }
}
