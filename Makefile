# Set the task name
TASK = acq_stat_reports

# set a version for the "make dist" option
VERSION = 2.1

# Uncomment the correct choice indicating either SKA or TST flight environment
FLIGHT_ENV = SKA


include /proj/sot/ska/include/Makefile.FLIGHT

SHARE = acq_stat_reports.py make_toc.pl acq_summarize.py star_error.py
TEMPLATES = templates/index.html templates/stars.html templates/summary.html
DATA = task_schedule.cfg acq_fail_fitfile.json

install:
ifdef TEMPLATES
	mkdir -p $(INSTALL_SHARE)/templates/
	rsync --times --cvs-exclude $(TEMPLATES) $(INSTALL_SHARE)/templates/
endif
ifdef SHARE
	mkdir -p $(INSTALL_SHARE)
	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/
endif
ifdef DATA
	mkdir -p $(INSTALL_DATA)
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/
endif

fit:
	python make_likelihood_fit.py
