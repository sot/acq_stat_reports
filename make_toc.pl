#!/usr/bin/env /proj/sot/ska/bin/perlska

use strict;
use warnings;
use IO::All;

my %opt = (  );

use Getopt::Long;
use Carp;

GetOptions( \%opt,
	    );


my $year_start = '2000';
my %type_expected = ( month => [qw( M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 )],
		      quarter => [qw( Q1 Q2 Q3 Q4 Q1)],
		      qspan => [qw( 1 3 3 3 2 )],
		      semi => [qw( S1 S2 S1)],
		      sspan => [qw( 1 6 5 )],
		      );
my %monthname =  ( M01 => 'Jan',
		   M02 => 'Feb',
		   M03 => 'Mar',
		   M04 => 'Apr',
		   M05 => 'May',
		   M06 => 'Jun',
		   M07 => 'Jul',
		   M08 => 'Aug',
		   M09 => 'Sep',
		   M10 => 'Oct',
		   M11 => 'Nov',
		   M12 => 'Dec',
		   );


my $task = 'guide_stat_reports';
my $SKA = $ENV{SKA} || '/proj/sot/ska';
my $SHARE = "${SKA}/share/guide_stat_db";
my $WEBDATA = "${SKA}/www/ASPECT/${task}";
my $SKADATA = "${SKA}/data/${task}";

#my $datafile = 'gs_report.yml';
my $webprefix = "http://cxc.harvard.edu/mta/ASPECT/${task}";
my $indexfile = 'index.html';

my %exist_dirs;

push @{$exist_dirs{month}}, map { $_ =~ s/${SKADATA}\///; $_  } glob("${SKADATA}/????/M[01]?");
push @{$exist_dirs{quarter}}, map { $_ =~ s/${SKADATA}\///; $_ } glob("${SKADATA}/????/Q?");
push @{$exist_dirs{semi}}, map { $_ =~ s/${SKADATA}\///; $_ } glob("${SKADATA}/????/S?");
push @{$exist_dirs{year}}, map { $_ =~ s/${SKADATA}\///; $_ } glob("${SKADATA}/????");

use Data::Dumper;
#print Dumper %exist_dirs;

my $toc;
$toc .= qq{ <HTML><HEAD><TITLE>Guide Statistics Reports</TITLE> \n} ;
$toc .= qq{ <link href="/mta/ASPECT/aspect.css" rel="stylesheet" type="text/css" media="all" /> \n};
$toc .= qq{    <style type="text/css"> \n };
$toc .= qq{ body { min-width:900px; background:url('http://asc.harvard.edu/mta/ASPECT/blue_paper.gif'); \n  }};
$toc .= qq{    </style> \n };
$toc .= qq{ </HEAD><BODY> };
$toc .= qq{ <H3>Guide Statistics Reports</H3> \n };
$toc .= qq{ <H4>Individual Reports</H4> \n};
$toc .= qq{ <TABLE BORDER=1> };
$toc .= qq{ <COLGROUP span=13 width="2*"></COLGROUP> };


for my $year ( $year_start ... $exist_dirs{year}->[-1] ){
#for my $year qw( 2000 ){
    $toc .= qq{<TR><TD rowspan=3><A HREF=\"${webprefix}/${year}/YEAR/${indexfile}\">$year</A></TD>};
    for my $month ( @{$type_expected{month}} ){
	my $string = "${year}/${month}";
	if (grep( /$string/, @{$exist_dirs{month}})){
	    $toc .= qq{ <TD><A HREF=\"${webprefix}/${year}/${month}/${indexfile}\">$monthname{$month}</A></TD> }
	}
	else{
	    $toc .= qq{ <TD>&nbsp;</TD> };
	}
	
    }
    $toc .= qq{</TR> \n};
    $toc .= qq{<TR>};
#	for my $quarter_idx  ( 0 ... $#{$type_expected{quarter}}){
    use POSIX;
    for my $quarter_idx  ( 0 ... 4){
	my $qyear = $year + floor($quarter_idx/4);
#	print "$quarter_idx $qyear \n";
	my $quarter = $type_expected{quarter}->[$quarter_idx];
	my $span = $type_expected{qspan}->[$quarter_idx];
	my $string = "${qyear}/${quarter}";
	my $quartertxt = qq{&nbsp;};
	if ( grep( /$string/, @{$exist_dirs{quarter}})){
	    $quartertxt = "<A HREF=\"${webprefix}/${qyear}/${quarter}/${indexfile}\">$qyear-$quarter</A>";
	}
	$toc .= qq{ <TD colspan=$span>$quartertxt</TD> };	
	
    }
    
    $toc .= qq{</TR> \n};
    $toc .= qq{<TR>};
#	for my $quarter_idx  ( 0 ... $#{$type_expected{quarter}}){
    use POSIX;
    for my $semi_idx  ( 0 ... 2){
	my $syear = $year + floor($semi_idx/2);
#	print "$quarter_idx $qyear \n";
	my $semi = $type_expected{semi}->[$semi_idx];
	my $span = $type_expected{sspan}->[$semi_idx];
	my $string = "${syear}/${semi}";
	my $txt = qq{&nbsp;};
	if ( grep( /$string/, @{$exist_dirs{semi}})){
	    $txt = "<A HREF=\"${webprefix}/${syear}/${semi}/${indexfile}\">$syear-$semi</A>";
	}
	$toc .= qq{ <TD colspan=$span>$txt</TD> };	
	
    }
    
    $toc .= qq{</TR> \n};

}

$toc .= qq{ </TABLE> \n};

$toc .= qq{ <H4>Summary Reports</H4> };
$toc .= qq{ <TABLE BORDER=1><TR> };
for my $type qw( Year Semi Quarter Month ){
    my $lctype = lc($type);
    $toc .= qq{ <TD><A HREF="${webprefix}/${lctype}_summary">By $type</A></TD> }
}
$toc .= qq{ </TR></TABLE> };


$toc .= qq{ <H4>Special Reports</H4> };
$toc .= qq{ <A HREF="${webprefix}/all_mission">Mission</A><BR />\n};
$toc .= qq{ <A HREF="${webprefix}/mission_since_2003">Mission Since 2003</A><BR />\n};

$toc .= qq{ </BODY></HTML> };

#print "$toc \n";



my $outfile = "index.html";
io("${WEBDATA}/${outfile}")->print($toc);



