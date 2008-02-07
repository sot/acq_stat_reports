<?

$web_base = "/mta/ASPECT/guide_stat_reports";
$gs_base = '/proj/sot/ska/www/ASPECT/guide_stat_reports';
$host  = $_SERVER['HTTP_HOST'];
$uri  = rtrim($_SERVER['PHP_SELF'], "/\\");
$script = $_SERVER['SCRIPT_NAME'];
// print "$uri vs $web_base vs $script";




<HTML>
<HEAD>
<TITLE>Guide Statistics Report</TITLE>
</HEAD>
<BODY>

<H2>Guide Statistics Report<!--#include virtual="title.htm"--></H2>
<H3><?php print file_get_contents("human_date_start.htm") ?> through
<?php print file_get_contents("human_date_stop.htm") ?> </H3>


<?php


//
//function show_overview(){
//}
//  
//
//function show_form( $year, $id ){
//  print '<FORM ACTION="folder.php" METHOD="get">';
//  print "Year <input type=\"text\" name=\"year\" value=\"$year\">";
//  print "ID <input type=\"text\" name=\"id\" value=\"$id\">";
//  print ' <input type="hidden" name="_submit_check" value="1"/>';
//  print ' <input type="submit" name="submit" value="Submit">';
//  print '<input type="reset" name="reset" value="Clear It">';
//  print '</FORM>';
//}
//
function show_nav( $year, $id){
  if (preg_match( '/^\d{2}$/', $id)){
    $nextid = $id + 1;
    $nextyear = $year;
    $previd = $id - 1;
    $prevyear = $year;
    if ($nextid == 13){
      $nextid = 1;
      $nextyear = $nextyear + 1;
    }
    if ($previd == 0){
      $previd = 12;
      $prevyear = $prevyear - 1;
    }
    //  print "$host $uri";
    $next = sprintf("http://${host}${uri}?year=%04d&id=%02d&_submit_check=1&submit=Submit", $nextyear, $nextid);
    $prev = sprintf("http://${host}${uri}?year=%04d&id=%02d&_submit_check=1&submit=Submit", $prevyear, $previd);
    print "<A HREF=\"$prev\">PREV</A>  <A HREF=\"$next\">NEXT</A>";
  }
}

show_nav( '2001', '01' )

//  
//
//
//}
//
//function process_form(){
//  //  show_form();
//  //  print $_POST[datestring];
//  show_form( $_GET[year], $_GET[id] );
//  show_nav( $_GET[year], $_GET[id] );
//  show_page( $_GET[year], $_GET[id] );
//  
//}
//
//function validate_form(){
//}
//
//
//if (array_key_exists('_submit_check',$_GET)) {
//  // If validate_form() returns errors, pass them to show_form()
//  if ($form_errors = validate_form()) {
//    show_form($form_errors);
//  } else {
//    // The submitted data is valid, so process it
//    process_form();
//  }
// } else {
//  // The form wasn't submitted, so display
//  show_form();
//  show_overview();
// }
//}

?>

<TABLE BORDER=1>
<TR><TH>TSTART</TH><TH>TSTOP</TH></TR>
<TR><TD><?php print file_get_contents("date_start.htm")?></TD><TD><?php print file_get_contents("date_stop.htm")?></TD></TR>
</TABLE>


<?php print file_get_contents("main_table.htm")?>


<TABLE>
<TD><?php print file_get_contents("histogram.htm")?></TD><TD><?php print file_get_contents("scatterplot.htm")?></TD>
</TABLE>

<?php print file_get_contents("mag_table.htm")?>

</BODY>
</HTML>



