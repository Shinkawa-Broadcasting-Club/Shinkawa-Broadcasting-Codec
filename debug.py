import sbcfast
import sbcfast.vs_arr
import sbcfast.
import vapoursynth as vs
core = vs.core
path = r"C:\Users\yyuuk\35_N71Tドラ_シャッター.mp4"
clip = core.lsmas.LWLibavSource(path)
planes = sbcfast.vs_arr.vs_to_np(clip)