use Cwd 'abs_path';
use File::Basename;

# Determine the project root
# If PROJECT_ROOT is set (by VS Code), use it. Otherwise, derive from this file's location.
my $root = $ENV{PROJECT_ROOT} || dirname(abs_path(__FILE__));

# Clean up path (remove redundant /../)
$root = abs_path($root);

$aux_dir = "$root/aux";
$out_dir = "$root/pdfs";

# LuaLaTeX configuration
$pdf_mode = 4;
$postscript_mode = $dvi_mode = 0;

# Ensure the engine itself knows where to put things
$lualatex = "lualatex -interaction=nonstopmode -shell-escape -synctex=1 -output-directory=\"$aux_dir\" %O %S";

# Ensure the auxiliary and output directories are found
$ENV{TEXINPUTS} = ".:$root/slides//:" . ($ENV{TEXINPUTS} || "");