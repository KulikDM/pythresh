REM rebuild docs shortcut
REM only works for Windows
cd..
xcopy examples\*.png docs\figs /Y
cd docs
call make clean
call make html