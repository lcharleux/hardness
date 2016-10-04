MODPATH = os.path.dirname(inspect.getfile(argiope))

def indentation_abqpostproc(workdir, path, odbPath, histPath, contactPath, fieldPath):
  """
  Writes the abqpostproc file in the workdir.
  """
  pattern = open(MODPATH + "/templates/indentation/abqpostproc.py").read()
  pattern = pattern.replace("#ODBPATH",     odbPath)
  pattern = pattern.replace("#HISTPATH",    histPath)
  pattern = pattern.replace("#CONTACTPATH", contactPath)
  pattern = pattern.replace("#FIELDPATH",   fieldPath)
  open(workdir + path, "wb").write(pattern)
      
def indentation_pypostproc(path, workdir, histPath, contactPath, fieldPath):
  """
  Writes the pypostproc file in the workdir.
  """
  pattern = open(MODPATH + "/templates/indentation/pypostproc.py").read() 
  pattern = pattern.replace("#HISTPATH",    histPath + ".rpt")
  pattern = pattern.replace("#CONTACTPATH", contactPath + ".rpt")
  pattern = pattern.replace("#FIELDPATH",   fieldPath + ".rpt")
  open(workdir + path, "wb").write(pattern)     
