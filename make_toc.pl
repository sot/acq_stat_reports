#!/usr/bin/env /proj/sot/ska/bin/perlska

use strict;
use warnings;
use IO::All;

my %opt = (  );

use Getopt::Long;
use Carp;

use POSIX;

GetOptions( \%opt,
	    );


my $year_start = '2000';
my %type_expected = ( month => [qw( M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 )],
		      quarter => [qw( Q1 Q2 Q3 Q4 Q1)],
		      semi => [qw( S1 S2 S1)],
		      year => [qw( YEAR )],
		      );

my %colspan = ( month => [qw( 1 1 1 1 1 1 1 1 1 1 1 1 )],
		quarter => [qw( 1 3 3 3 2 )],
		semi => [qw( 1 6 5 )],
		year => [qw( 12 )],
		);

my %n_intervals = ( month => 12,
		    quarter => 4,
		    semi => 2,
		    year => 1 );

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


my $task = 'acq_stat_reports';
my $SKA = $ENV{SKA} || '/proj/sot/ska';
my $SHARE = "${SKA}/share/${task}";
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


#print Dumper %exist_dirs;

my $toc;
$toc .= qq{ <HTML><HEAD><TITLE>Acquisition Statistics Reports</TITLE> \n} ;
$toc .= qq{ <link href="/mta/ASPECT/aspect.css" rel="stylesheet" type="text/css" media="all" /> \n};
$toc .= qq{    <style type="text/css"> \n };
$toc .= qq{ body { min-width:900px; background:url('http://asc.harvard.edu/mta/ASPECT/blue_paper.gif'); \n  }};
$toc .= qq{    </style> \n };
$toc .= qq{ </HEAD><BODY> };
$toc .= qq{ <H3>Acquisition Statistics Reports</H3> \n };
$toc .= qq{ <H4>Individual Reports</H4> \n};


for my $typestring qw( Month Quarter Semi Year ){

    $toc .=  "<P>${typestring} Data</P>";

    $toc .= qq{ <TABLE BORDER=1> };
    $toc .= qq{ <COLGROUP span=13 width="2*"></COLGROUP> };
   
    $toc .= qq{<TR><TD></TD>};
    for my $month ( @{$type_expected{month}}){
	$toc .= qq{ <TD>$monthname{$month}</TD> }
    }
    $toc .= qq{</TR> \n};
    $toc .= qq{<TR>};


    for my $year ( $year_start ... $exist_dirs{year}->[-1] ){
	$toc .= qq{<TR><TD><A HREF=\"${webprefix}/${year}/YEAR/${indexfile}\">$year</A></TD>};
	my $type = lc($typestring);
	my $interval_count = 0;
	for my $interval (@{$type_expected{$type}}){
	    my $text = ( $type =~ /month/ ) ? $monthname{$interval} 
                     : ( $type =~ /year/ )  ? $year 
		     :                              $interval ;
	    my $interval_year = ( $type =~ /month/ ) ? $year 
                              : ( $type =~ /year/ ) ? $year
                              : ( $year + floor( $interval_count / $n_intervals{$type} )) ;
	    my $string = "${interval_year}/${interval}";

	    my $interval_colspan = $colspan{$type}->[$interval_count];
	    print "$interval_year $interval $string $interval_colspan\n";
	    if (grep( /$string/, @{$exist_dirs{$type}})){
		$toc .= qq{ <TD colspan=${interval_colspan}><A HREF=\"${webprefix}/${interval_year}/${interval}/${indexfile}\">$text</A></TD> }
	    }
	    else{
		$toc .= qq{ <TD colspan=${interval_colspan}>&nbsp;</TD> };
	    }

	    $interval_count++;
	}
	$toc .= qq{</TR> \n};

    }

    $toc .= qq{ </TABLE> \n};
}




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


my $outfile = "index.html";
io("${WEBDATA}/${outfile}")->print($toc);



