#!/usr/bin/env /proj/sot/ska/bin/perlska

use strict;
use warnings;

use YAML;
use Data::Dumper;

my %opt = (  );

use Getopt::Long;
use Carp;

GetOptions( \%opt,
	    'month!',
	    'quarter!',
	    'semi!',
	    'year!',
	    );


use Chandra::Time;

my $year_start = '2000';
my %type_expected = ( month => [qw( M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 )],
		      quarter => [qw( Q1 Q2 Q3 Q4 )],
		      semi => [qw( S1 S2 )],
		      year => [qw( YEAR )],
		      );


my $task = 'guide_stat_reports';
my $SKA = $ENV{SKA} || '/proj/sot/ska';
my $SHARE = "${SKA}/share/guide_stat_db";
my $WEBDATA = "${SKA}/www/ASPECT/${task}";
my $SKADATA = "${SKA}/data/${task}";

my $datafile = 'gs_report.yml';
my $webprefix = "http://icxc.harvard.edu/ska/guidestats/";

my @exist_dirs;
my $title;
my $outdir;
my $type;
if ($opt{month}){
    @exist_dirs = glob("${SKADATA}/????/M[01]?");
    $title = 'Month';
    $outdir = "${WEBDATA}/month_summary/";
    $type = 'month';
}
if ($opt{quarter}){
    @exist_dirs = glob("${SKADATA}/????/Q?");
    $title = 'Quarter';
    $outdir = "${WEBDATA}/quarter_summary/";
    $type = 'quarter';
}
if ($opt{semi}){
    @exist_dirs = glob("${SKADATA}/????/S?");
    $title = 'Semi-Annual';
    $outdir = "${WEBDATA}/semi_summary/";
    $type = 'semi';
}
if ($opt{year}){
    @exist_dirs = glob("${SKADATA}/????/YEAR");
    $title = 'Year';
    $outdir = "${WEBDATA}/year_summary/";
    $type = 'year';
}


if (not defined $title){
    croak('Specify type of summary as option: -month, -quarter, -semi, or -year');
}

my $outfile = "index.html";

use File::Path;
mkpath( $outdir, 1);

my $starfile = 'bad_stars.gif';
my $ratefile = 'bad_rate.gif';
my %plots;
my $table;

$table .= qq{ <HTML><HEAD><TITLE>$title Summary</TITLE> };
$table .= qq^ <link href="/mta/ASPECT/aspect.css" rel="stylesheet" type="text/css" media="all" /> ^;
$table .= qq^ <style type="text/css"> body { min-width:900px; background:url('http://asc.harvard.edu/mta/ASPECT/blue_paper.gif'); } } </style> ^;
$table .= qq{ </HEAD><BODY> };
$table .= qq{ <H3>$title Summary</H3> \n };
$table .= qq{ <TABLE> };
$table .= qq{ <TR><TD><IMG SRC="./bad_stars.gif"></TD></TR> };
$table .= qq{ </TABLE> };

$table .= qq{ <TABLE BORDER=1> };
$table .= qq{ <TR><TH>DirNum</TH><TH>$title</TH><TH>$title Bad Rate</TH><TH>Mission Bad Rate</TH><TH>$title Failed Rate</TH><TH>Mission Failed Rate</TH><TH>$title OBC Bad Status</TH><TH>Mission OBC Bad Status</TH></TR> };

# Now that I'm not making reports for no-data intervals, I've got to figure out how many dirs I should have
my @dirs;
my $lastdir = $exist_dirs[-1];
$lastdir =~ s/${SKADATA}\///;
$lastdir =~ /^(\d{4})\/(\S+)$/;
my $last_year = $1;
my $last_id = $2;
my @expected = @{$type_expected{$type}};

for my $year ( $year_start ... $last_year ){
    for my $id (@expected){
	push @dirs, "${year}/${id}";
	last if ( ($year == $last_year) and ($id eq $last_id ));
    }	
}

for my $idx (0 ... $#dirs){
    my $dirname = $dirs[$idx];
    my $dirabbrev = $dirname;
    $dirabbrev =~ s/\//-/g;
    push @{$plots{dirnum}}, $idx;

    if (-e "${SKADATA}/${dirname}/${datafile}"){
	print "report for $dirname \n";
	my %data = %{YAML::LoadFile("${SKADATA}/${dirname}/${datafile}")};
	$table .= qq{ <TR><TD>$idx</TD><TD><A HREF="${webprefix}/${dirname}">$dirabbrev</A></TD> };
#    my $dirabbrev = $dirname;
#	push @{$plots{dir_id}}, $dirabbrev;

#	push @{$plots{webdir}}, $dirname;
#    push @{$plots{bad_track_stars}}, $data{DATA}->{report}->{bad_track_stars};
	
	push @{$plots{bad_track_rate}}, $data{DATA}->{report}->{bad_track_rate};
	push @{$plots{mission_bad_track_rate}}, $data{DATA}->{mission}->{bad_track_rate};
	push @{$plots{obc_bad_status_rate}}, $data{DATA}->{report}->{obc_bad_status_rate};
	push @{$plots{mission_obc_bad_status_rate}}, $data{DATA}->{mission}->{obc_bad_status_rate};
	push @{$plots{fail_track_rate}}, $data{DATA}->{report}->{fail_track_rate};
	push @{$plots{mission_fail_track_rate}}, $data{DATA}->{mission}->{fail_track_rate};
	$table .= sprintf("<TD>%6.3f</TD>", $data{DATA}->{report}->{bad_track_rate});
	$table .= sprintf("<TD>%6.3f</TD>", $data{DATA}->{mission}->{bad_track_rate});
	$table .= sprintf("<TD>%6.3f</TD>", $data{DATA}->{report}->{fail_track_rate});
	$table .= sprintf("<TD>%6.3f</TD>", $data{DATA}->{mission}->{fail_track_rate});
	$table .= sprintf("<TD>%6.3f</TD>", $data{DATA}->{report}->{obc_bad_status_rate});
	$table .= sprintf("<TD>%6.3f</TD>", $data{DATA}->{mission}->{obc_bad_status_rate});

    }
    else{
	$table .= qq{ <TR><TD>$idx</TD><TD>$dirabbrev</TD> };
	print "No data for $dirname \n";
	push @{$plots{bad_track_rate}}, undef;
	push @{$plots{mission_bad_track_rate}}, undef;
	push @{$plots{obc_bad_status_rate}}, undef;
	push @{$plots{mission_obc_bad_status_rate}}, undef;
	push @{$plots{fail_track_rate}}, undef;
	push @{$plots{mission_fail_track_rate}}, undef;
	$table .= sprintf("<TD colspan=6></TD>");
    }
    $table .= qq{ </TR> };
}

$table .= qq{ </TABLE></BODY></HTML> };

use IO::All;
my $out = io("${outdir}/${outfile}");
$out->print($table);

use PGPLOT::Simple qw( pgs_plot );
use PDL;

my @plot1 = (
	    'x' => pdl(@{$plots{dirnum}}),
	    'y' => pdl(@{$plots{bad_track_rate}})*100,
	    panel => [1,1],
#	    logy => 1,
#	    lims => [$mag_start,$mag_stop, 0.2, undef],
	    options => {center => 1},
	    charsize => {symbol => 0.7,
			 title => 2.0,
			 axis => 2.0,
		     },
	    toptitle => "Bad Star Rate",
	    xtitle => "$title from 2000-01",
	    ytitle => "Bad Track Rate x100 (Red = Mission )",
#	    toptitle => "Mags for good (black) and bad (red) guide stars",
#	    xtitle => 'Star magnitude (mag)',
#	    ytitle => 'Number (red is x100)',
#	    @label,
	    plot => 'bin',
	    'x' => pdl(@{$plots{dirnum}})+0.1,
	    'y' => pdl(@{$plots{mission_bad_track_rate}})*100,
	    color => { line => 'red' },
	    plot => 'bin',
	    );

my @plot2 = (
	    'x' => pdl(@{$plots{dirnum}}),
	    'y' => pdl(@{$plots{fail_track_rate}})*100,
	    panel => [2,1],
#	    logy => 1,
#	    lims => [$mag_start,$mag_stop, 0.2, undef],
	    options => {center => 1},
	    charsize => {symbol => 0.7,
			 title => 2.0,
			 axis => 2.0,
		     },
	    toptitle => "Fail Star Rate",
	    xtitle => "$title from 2000-01",
	    ytitle => "Fail Rate x100 (Red = Mission )",
#	    toptitle => "Mags for good (black) and bad (red) guide stars",
#	    xtitle => 'Star magnitude (mag)',
#	    ytitle => 'Number (red is x100)',
#	    @label,
	    plot => 'bin',
	    'x' => pdl(@{$plots{dirnum}})+0.1,
	    'y' => pdl(@{$plots{mission_fail_track_rate}})*100,
	    color => { line => 'red' },
	    plot => 'bin',

#	    'x' => pdl(@mag_bin)+0.1,
#	    'y' => pdl(@bad_track_x100)+.01,
#	    color => { line => 'red' },
#	    plot => 'bin',
	    );


my @plot3 = (
	    'x' => pdl(@{$plots{dirnum}}),
	    'y' => pdl(@{$plots{obc_bad_status_rate}})*100,
	    panel => [3,1],
#	    logy => 1,
#	    lims => [$mag_start,$mag_stop, 0.2, undef],
	    options => {center => 1},
	    charsize => {symbol => 0.7,
			 title => 2.0,
			 axis => 2.0,
		     },
	    toptitle => "OBC Bad Status Rate",
	    xtitle => "$title from 2000-01",
	    ytitle => "OBC Bad Rate x100 (Red = Mission)",
#	    toptitle => "Mags for good (black) and bad (red) guide stars",
#	    xtitle => 'Star magnitude (mag)',
#	    ytitle => 'Number (red is x100)',
#	    @label,
	    plot => 'bin',
	    'x' => pdl(@{$plots{dirnum}})+0.1,
	    'y' => pdl(@{$plots{mission_obc_bad_status_rate}})*100,
	    color => { line => 'red' },
	    plot => 'bin',

#	    'x' => pdl(@mag_bin)+0.1,
#	    'y' => pdl(@bad_track_x100)+.01,
#	    color => { line => 'red' },
#	    plot => 'bin',
	    );



#print Dumper %plots;
#$file = $save_prefix . $file;
my $file = $starfile;
$file =~ s/\.gif/\.ps\/vcps/;
$file = "${outdir}/" . $file;
pgs_plot(
	 nx => 3,
	 ny => 1,
	 xsize => 12,
	 ysize => 4,
	 device => $file,
	 @plot1,
	 @plot2,
	 @plot3,
	 );
$file =~ s/\/vcps$//;
my $psfile = $file;
$file =~ s/\.ps/.gif/;
use Ska::Run;
run("convert -antialias $psfile $file");

