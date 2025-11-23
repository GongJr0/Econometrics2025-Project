import os
import argparse
from pdflatex import PDFLaTeX

def export_latex(filename: str, out_dir: str, verbose=False):
    pdfl = PDFLaTeX.from_texfile(filename)

    pdf, log, cp = pdfl.create_pdf(keep_pdf_file=False, keep_log_file=False)
    if out_dir == '':
        out_dir = 'out'

    os.makedirs(out_dir, exist_ok=True)
    if cp.returncode == 0: # Success    
        outfile = os.path.join(out_dir, f"{filename}.pdf")
        with open(outfile, 'wb') as f:
            f.write(pdf)
        if verbose:
            print(log)
    
    else: # Fail
        outlog = os.path.join(out_dir, f"{filename}_error.log")
        with open(outlog, 'wb') as f:
            f.write(log)
        if verbose:
            print(log)
    return cp.returncode  # Return error of LaTeX compiler did so


def main() -> int:
    parser = argparse.ArgumentParser('LaTeXport')
    parser.add_argument('filename', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default="")
    parser.add_argument('-v', '--verbose', type=bool, default=False)

    args = parser.parse_args()
    code = export_latex(args.filename, args.output_dir, args.verbose)
    return code