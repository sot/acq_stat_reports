# Set the task name
TASK = acq_stat_reports
PERLTASK = AcqStats

# set a version for the "make dist" option
VERSION = 1.0

# Uncomment the correct choice indicating either SKA or TST flight environment
FLIGHT_ENV = SKA
# FLIGHT_ENV = TST

include /proj/sot/ska/include/Makefile.FLIGHT


#ICXC_CGI_DIR = /proj/web-icxc/cgi-bin/aspect/
#CGI_DIR = ${ICXC_CGI_DIR}/${TASK}
#CGI_TEST_DIR = ${CGI_DIR}_test
#WEB_DIR = /proj/sot/ska/www/ASPECT/${TASK}
#WEB_TEST_DIR = ${WEB_DIR}_test


# Set the names of all files that get installed
#  Examples for celmon
#  TASK = celmon
#  BIN = celmon.pl 
#  SHARE = calc_offset.pl
#  DATA = CELMON_table.rdb ICRS_tables

#CGI = guide_stat_web_query.cgi show_month_stats.cgi
#DATA = 
SHARE = make_report.pl make_summary.pl make_toc.pl report.yaml standard_report.html all_plot_report.html
#PROJLIB = Report.pm


# Define outside data and bin dependencies required for testing,
# i.e. all tools and data required by the task which are NOT 
# created by or internal to the task itself.  These will be copied
# from the ROOT_FLIGHT area.
#
# TEST_DEP = bin/skycoor data/EPHEM/gephem.dat

#TEST_DEP = bin/sysarch bin/syspathsubst share/telem_archive/fetch.py

#install_dirs:
#	mkdir -pv $(INSTALL_BIN)
#	mkdir -pv $(INSTALL_SHARE)



# To 'test', first check that the INSTALL root is not the same as the FLIGHT root
# with 'check_install' (defined in Makefile.FLIGHT).  Typically this means doing
#  setenv TST $PWD
# Then copy any outside data or bin dependencies into local directory via
# dependency rules defined in Makefile.FLIGHT

# Testing no long creates a lib/perl link, since Perl should find the library
# because perlska puts /proj/sot/ska/lib/perl (hardwired) into PERL5LIB.

#test: check_install install_dirs $(TEST_DEP) install
#	$(INSTALL_SHARE)/update_guide_stats_db.pl -dryrun -obsid 5

# An example of compiling a fortran program which uses a library
#cocoxmm: cocoxmm.f
#	f77 cocoxmm.f -L$(ROOT_FLIGHT)/lib -lGEOPACK -o cocoxmm




dist:
	mkdir $(TASK)-$(VERSION)
	rsync -aruvz --cvs-exclude --exclude $(TASK)-$(VERSION) * $(TASK)-$(VERSION)
	tar cvf $(TASK)-$(VERSION).tar $(TASK)-$(VERSION)
	gzip --best $(TASK)-$(VERSION).tar
	rm -rf $(TASK)-$(VERSION)/


#WEB = index.html

inv_check_install:
	test "$(INSTALL)" = "$(ROOT_FLIGHT)"		

test: check_install $(BIN) $(CGI) $(LIB)
ifdef BIN
	mkdir -p $(INSTALL_BIN)
	rsync --times --cvs-exclude $(BIN) $(INSTALL_BIN)/
endif
ifdef DATA
	mkdir -p $(INSTALL_DATA)
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/
endif
ifdef CGI
	mkdir -p $(CGI_TEST_DIR)
	rsync --times --cvs-exclude $(CGI) $(CGI_TEST_DIR)/
endif
ifdef WEB
	mkdir -p $(WEB_TEST_DIR)
	sed -e "s/aspect\/$(TASK)/aspect\/$(TASK)_test/g" $(WEB) > $(WEB_TEST_DIR)/$(WEB)
#	rsync --times --cvs-exclude $(WEB) $(WEB_TEST_DIR)/
endif
ifdef LIB
	mkdir -p $(INSTALL_PERLLIB)
	rsync --times --cvs-exclude $(LIB) $(INSTALL_PERLLIB)/
endif

delete_test:
	if [ -r $(WEB_TEST_DIR) ] ; then rm -r $(WEB_TEST_DIR); fi
	if [ -r $(CGI_TEST_DIR) ] ; then rm -r $(CGI_TEST_DIR); fi


install: inv_check_install 
ifdef SHARE
	mkdir -p $(INSTALL_SHARE)
	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/
endif
ifdef BIN
	mkdir -p $(INSTALL_BIN)
	rsync --times --cvs-exclude $(BIN) $(INSTALL_BIN)/
endif
ifdef DATA
	mkdir -p $(INSTALL_DATA)
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/
endif
ifdef CGI
	mkdir -p $(CGI_DIR)
	rsync --times --cvs-exclude $(CGI) $(CGI_DIR)/
endif
ifdef WEB
	mkdir -p $(WEB_DIR)
	rsync --times --cvs-exclude $(WEB) $(WEB_DIR)/
endif
ifdef LIB
	mkdir -p $(INSTALL_PERLLIB)/Ska/
	rsync --times --cvs-exclude $(LIB) $(INSTALL_PERLLIB)/Ska/
endif
ifdef PROJLIB
	mkdir -p $(INSTALL_PERLLIB)/Ska/$(PERLTASK)/
	rsync --times --cvs-exclude $(PROJLIB) $(INSTALL_PERLLIB)/Ska/$(PERLTASK)/
endif

web_install: 
ifdef CGI
	mkdir -p $(CGI_DIR)
	rsync --times --cvs-exclude $(CGI) $(CGI_DIR)/
endif
ifdef WEB
	mkdir -p $(WEB_DIR)
	rsync --times --cvs-exclude $(WEB) $(WEB_DIR)/
endif
