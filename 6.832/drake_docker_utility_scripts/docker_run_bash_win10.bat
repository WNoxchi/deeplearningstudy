@ECHO "Drake Docker container for WINDOWS"
@if "%3" == "" goto args_count_wrong
@if "%4" == "" goto args_count_ok 

:args_count_wrong
@ECHO "Please run from the command line and "
@ECHO " supply three arguments: a Drake release (drake-YYYYMMDD),"
@ECHO " a relative path to a directory to mount as /notebooks, and your"
@ECHO " computer's IP address (from running e.g. ipconfig)."
@PAUSE
@exit /b 1

:args_count_ok
docker pull mit6832/drake-course:%1
docker run -it -e DISPLAY=%3:0 --rm -v "%cd%\%2":/notebooks mit6832/drake-course:%1 /bin/bash -c "cd /notebooks && /bin/bash"
