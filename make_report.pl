#!/usr/bin/env /proj/sot/ska/bin/perlska

use strict;
use warnings;

use Data::Dumper;
use Getopt::Long;
use IO::All;
use YAML;

use Date::Format;
use Chandra::Time;
use Date::Tie;

use Ska::Report::TimeRange;
use Pod::Help qw( -help);

#use Ska::AcqStats::Report;
use Ska::StarStats::Report;
#require "Report.pm";

=pod

=head1 NAME

make_report.pl - script to extract guide star statistics for a specified interval

=head1 SYNOPSIS

B<make_report.pl> [I<options>]
 
=cut


my $task = 'acq_stat_reports';
my $SKA = $ENV{SKA} || '/proj/sot/ska';
my $SHARE = "${SKA}/share/${task}";
my $SKADATA = "${SKA}/data/${task}";

my %config;
if (-e "report.yaml"){
    %config = YAML::LoadFile("report.yaml");
}
else{
    if (-e "${SHARE}/report.yaml"){
	%config = YAML::LoadFile("${SHARE}/report.yaml");
    }
    else{
	die("No report.yaml config file at ${SHARE}/report.yaml");
    }
}

# Set up default times
my $now_time = Date::Tie->new();
my $mission_start = $config{task}->{mission_start_time};
my $start_time = Date::Tie->new( year => $mission_start->{year}, 
				 month => $mission_start->{month}, 
				 day => $mission_start->{day} );

my %whole_mission_interval = ( start => $start_time,
			       stop => $now_time );

# for default behavior of previous month time range
my $month_back_time = $now_time->new();
$month_back_time->{month}--;

my %default_interval = ( start => $month_back_time,
			 stop => $now_time );


#print Ska::AcqStats::Report::date_to_fits_format( $start_time ), "\n";
#print Ska::AcqStats::Report::date_to_fits_format( $now_time ), "\n";


my %opt = ( calc_rate_tstart => Ska::Report::TimeRange::datetie_to_fits_format($start_time),
	    calc_rate_tstop => Ska::Report::TimeRange::datetie_to_fits_format($now_time),
	    tstart => Ska::Report::TimeRange::datetie_to_fits_format($month_back_time),
	    tstop => Ska::Report::TimeRange::datetie_to_fits_format($now_time),
	    print_web => 1,
	    );

# retrieve tool requested times

GetOptions( \%opt,
	    "dryrun!",
	    "tstart=s",
	    "tstop=s",
	    "predefined=s",
	    "outdir=s",
	    "update!",
	    "no_summary!",
	    );


#

my %quarter_end_month = ( 1 => 1, # quarter 1, Nov -> Jan
			2 => 4, # quarter 2, Feb -> April
			3 => 7, # quarter 3, May -> July
			4 => 10, # quarter 4, Aug -> Oct
			);
my %semi_end_month = ( 1 => 1 , # semi Aug -> January
		       2 => 7, # semi Feb -> July
		       );
		    
my %end_month = ( quarter => \%quarter_end_month,
		  semi => \%semi_end_month );

if ( defined $opt{predefined} ){
    run_predefined( { config => \%config, opt => \%opt });
    exit;
}


    
if (defined $opt{update}){
    my @intervals = qw( month quarter semi year );
    for my $type (@intervals){
	my $interval = Ska::Report::TimeRange->new( $type, { out_fmt => 'datetie'} )->find_prev();
	my $range_ref = $interval->range();
	my $label = $range_ref->{label};
	$opt{predefined} = $label;
	$label =~ s/-/\//g;
	if (-e "${SKADATA}/${label}" ){
#	    print "${label} - Up to date. \n";
	}
	else{
#	    print "${label} - Updating ... \n";
	    run_predefined( { config => \%config, opt => \%opt });
	}
    }
    exit;
}
						
if (defined $opt{outdir}){
    $opt{save_path} = $opt{outdir};
}


run_report({ config => \%config, opt => \%opt });    


sub run_report{

    my $arg_in = shift;
    my %config = %{$arg_in->{config}};
    my %opt = %{$arg_in->{opt}};


    eval{
	$config{task}->{template_file} = 'index.php';
	$config{task}->{report_file} = 'index.php';
	Ska::StarStats::Report->new({ 
	    task => 'acq_stat_reports',
	    config => \%config,
	    opt => \%opt,
	})->standard_report();
	
    };
    if ($@){
	print "$@\n";
    }

}


sub run_predefined{
    my $arg_in = shift;
    my %config = %{$arg_in->{config}};
    my %opt = %{$arg_in->{opt}};

    print "Updating ", $opt{predefined}, " ";

    my $month_start;
    my $month_end;
    my $type;
    # yearly
    if ( $opt{predefined} =~ /^(\d{4})-YEAR$/i ){
	my $year = $1;
	$opt{year} = $year;
	$opt{id} = 'YEAR';
	$opt{title} = " - Yearly Report : $year ";
	$month_start = Date::Tie->new( year => $year, month => 1,
				       day => 1, hour => 0, minute => 0, second => 0);
	$month_end = $month_start->new();
	$month_end->{month} = 12;
	$type = 'year';
    }
    # semi
    if ( $opt{predefined} =~ /^(\d{4})-[sS]([12])$/){
	my $year = $1;
	my $semi = $2;
	$opt{year} = $year;
	$opt{id} = "S${semi}";
	$month_end = Date::Tie->new( year => $year, month => $end_month{semi}->{$semi},
				     day => 1, hour => 0, minute => 0, second => 0 );
	$month_start = $month_end->new();
	$month_start->{month}-= 5;
	my $unix_t_start = Chandra::Time->new( Ska::Report::TimeRange::datetie_to_doy_format($month_start))->unix();
	my $unix_t_end = Chandra::Time->new( Ska::Report::TimeRange::datetie_to_doy_format($month_end))->unix();
	my $month_start_string = time2str("%B", $unix_t_start, '+0000' );
	my $month_end_string = time2str("%B", $unix_t_end, '+0000' );
	$opt{title} = " - Semi-Annual Report, $year, $month_start_string through $month_end_string";	
	$type = 'semi';
    }
    # quarterly
    if ( $opt{predefined} =~ /^(\d{4})-[qQ]([1234])$/ ){
	my $year = $1;
	my $quarter = $2;
	$opt{year} = $year;
	$opt{id} = "Q${quarter}";
	$month_end = Date::Tie->new( year => $year, month => $end_month{quarter}->{$quarter},
				     day => 1, hour => 0, minute => 0, second => 0 );
	$month_start = $month_end->new();
	$month_start->{month}-= 2;
	my $unix_t_start = Chandra::Time->new( Ska::Report::TimeRange::datetie_to_doy_format($month_start))->unix();
	my $unix_t_end = Chandra::Time->new( Ska::Report::TimeRange::datetie_to_doy_format($month_end))->unix();
	my $month_start_string = time2str("%B", $unix_t_start, '+0000' );
	my $month_end_string = time2str("%B", $unix_t_end, '+0000' );
	$opt{title} = " - Quarterly Report, $year, $month_start_string through $month_end_string";	
	$type = 'quarter';
    }
    # monthly
    if ( $opt{predefined} =~ /^(\d{4})-[mM](\d{2})$/ ){
	my $year = $1;
	my $month = $2;
	$opt{year} = $year;
	$opt{id} = "M${month}";
	$month_end = Date::Tie->new( year => $year, month => $month,
				     day => 1, minute => 0, hour => 0, second => 0 );
	$month_start = $month_end->new();
	my $unix_t_start = Chandra::Time->new( Ska::Report::TimeRange::datetie_to_doy_format($month_start))->unix();
	my $month_start_string = time2str("%B", $unix_t_start, '+0000' );
	$opt{title} = " - Monthly Report, $year, $month_start_string";	
	$type = 'month';
    }
    unless (defined $type){
	die( "No format matched ");
    }

    # set month end to be the end of its month
    $month_end->{month}++;
    $month_end->{second}--;

#    print Ska::Report::TimeRange::datetie_to_fits_format( $month_start ), "\n";
#    print Ska::Report::TimeRange::datetie_to_fits_format( $month_end ), "\n";
    
    $opt{save_path} = $opt{year} . "/" . $opt{id} . "/" ;
    
    $opt{tstart} =  Ska::Report::TimeRange::datetie_to_fits_format( $month_start );
    $opt{tstop} =  Ska::Report::TimeRange::datetie_to_fits_format( $month_end );
    $opt{calc_rate_tstop} = $opt{tstop};

    run_report({ config => \%config, opt => \%opt });    
    use Ska::Run;
    unless (defined $opt{no_summary}){
	run("${SHARE}/make_summary.pl -${type}");
    }
    run("${SHARE}/make_toc.pl");

}

=pod

=head1 OPTIONS

=over 4

=item B<-help>

Print this help information

=item B<-tstart> time (in Chandra::Time recognized format )

Set the beginning of a time interval upon which to report

=item B<-tstop> time (in Chandra::Time recognized format )

Set the end of a time interval upon which to report

=item B<-predefined> month, quarter, semiannual, or year ( 2005-M12, 2007-Q2, 2005-S2, 2001-YEAR )

Overrides -tstart and -tstop with a defined interval of the recognized types
Runs the summary tool as well to update the summary for the type.

=item B<-update>

Uses Ska::Report::TimeRange to find the most recent complete time ranges and checks to see if those
predefined ranges have been processed.  Updates if needed. 

=item B<-no_summary>

Don't update the appropriate summary file if running in predefined mode

=back

=head1 DESCRIPTION


=cut




