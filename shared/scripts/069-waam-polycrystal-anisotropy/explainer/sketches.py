import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
NAVY="#1F3864"; RED="#c0392b"; BLUE="#2c5aa0"; GREY="0.5"

# ---------- A: cubic unit cell + <100> axes (+ directional stiffness) ----------
fig=plt.figure(figsize=(6.2,5.2)); ax=fig.add_subplot(111,projection="3d")
r=[0,1]; import itertools
pts=np.array(list(itertools.product(r,r,r)))
edges=[(a,b) for a in range(8) for b in range(8)
       if np.sum(np.abs(pts[a]-pts[b]))==1 and a<b]
segs=[[pts[a],pts[b]] for a,b in edges]
ax.add_collection3d(Line3DCollection(segs,colors="0.55",lw=1.4))
# atoms: FCC corners + face centres
corners=pts
faces=np.array([[.5,.5,0],[.5,.5,1],[.5,0,.5],[.5,1,.5],[0,.5,.5],[1,.5,.5]])
ax.scatter(*corners.T,s=55,c=NAVY,depthshade=False)
ax.scatter(*faces.T,s=55,c="#7f9bc0",depthshade=False)
def arr(ax,v,txt,c):
    ax.quiver(0,0,0,*v,color=c,lw=2.6,arrow_length_ratio=0.13)
    ax.text(*(np.array(v)*1.08),txt,color=c,fontsize=12,weight="bold")
arr(ax,[1.28,0,0],"[100]",RED); arr(ax,[0,1.28,0],"[010]",RED); arr(ax,[0,0,1.28],"[001]",RED)
ax.plot([0,1],[0,1],[0,1],color=BLUE,lw=2.2,ls="--")
ax.text(1.0,1.0,1.08,"[111]",color=BLUE,fontsize=11)
ax.text2D(0.5,1.02,"Kubische Elementarzelle (FCC, Austenit) — die drei ⟨100⟩-Achsen",
          transform=ax.transAxes,ha="center",fontsize=11,color=NAVY)
ax.text2D(0.5,-0.02,"⟨100⟩ = Würfelkanten (weichste Richtung, E≈94 GPa)   ·   [111] = Raumdiagonale (steifste, E≈300 GPa)",
          transform=ax.transAxes,ha="center",fontsize=8.5,color="0.3")
ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(22,35)
ax.set_xlim3d(-0.25,1.45); ax.set_ylim3d(-0.25,1.45); ax.set_zlim3d(-0.25,1.45)
fig.tight_layout(); fig.savefig("expl_A_cube.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- B: grains with lattice orientation: random vs textured ----------
fig,axs=plt.subplots(1,2,figsize=(11,5))
rng=np.random.default_rng(3)
# a few grain polygons via simple Voronoi-like: use fixed hand polygons
def grain_cells(ax):
    # tile the unit square into ~6 irregular cells (precomputed seeds Voronoi)
    from scipy.spatial import Voronoi
    seeds=np.array([[.25,.25],[.7,.2],[.8,.65],[.45,.55],[.2,.75],[.55,.85]])
    # clip Voronoi manually is messy; instead draw seeds as grain centres + boxes
    return seeds
for ax,(title,mode) in zip(axs,[("regellose Textur (isotrop)","rand"),
                                 ("scharfe ⟨100⟩-Textur (⟨100⟩ ∥ Aufbau & Schweiß)","tex")]):
    ax.add_patch(Rectangle((0,0),1,1,fill=False,ec="0.3",lw=1.5))
    seeds=np.array([[.24,.26],[.7,.22],[.8,.66],[.46,.55],[.2,.78],[.6,.85],[.83,.4],[.4,.15]])
    for (cx,cy) in seeds:
        ang = rng.uniform(0,90) if mode=="rand" else rng.normal(0,7)
        a=np.deg2rad(ang); R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        s=0.085
        sq=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*s
        sq=(R@sq.T).T+[cx,cy]
        ax.add_patch(Polygon(sq,closed=True,fc="#dbe4f0",ec=NAVY,lw=1.3))
        # one <100> axis as arrow
        d=(R@np.array([1,0]))*s*1.7
        ax.annotate("",xy=(cx+d[0],cy+d[1]),xytext=(cx,cy),
                    arrowprops=dict(arrowstyle="->",color=RED,lw=1.6))
    ax.annotate("",xy=(0.0,1.08),xytext=(0.0,0.0),arrowprops=dict(arrowstyle="->",color="0.4",lw=1.4),
                annotation_clip=False)
    ax.text(-0.02,1.10,"Aufbau",fontsize=8,color="0.4",ha="right")
    ax.annotate("",xy=(1.08,0.0),xytext=(0.0,0.0),arrowprops=dict(arrowstyle="->",color="0.4",lw=1.4),
                annotation_clip=False)
    ax.text(1.10,-0.02,"Schweiß",fontsize=8,color="0.4",va="top")
    ax.set_title(title,fontsize=11,color=NAVY); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.15,1.2); ax.set_ylim(-0.15,1.2)
fig.suptitle("Körner = kleine Kristalle mit je eigener Gitter-Orientierung (roter Pfeil = eine ⟨100⟩-Achse)",
             fontsize=11.5)
fig.tight_layout(); fig.savefig("expl_B_grains.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- C: stereographic projection idea ----------
fig,ax=plt.subplots(figsize=(6.6,4.6))
th=np.linspace(0,np.pi,200)
ax.plot(np.cos(th),np.sin(th),color="0.4",lw=1.5)          # hemisphere (dome)
ax.plot([-1.15,1.15],[0,0],color="0.4",lw=1.2)             # equator plane (the disc, edge-on)
ax.plot(0,0,"k+",ms=9,mew=1.6)
for (a,lab,c) in [(75,"fast ∥ Normale\n→ nahe Zentrum",RED),(25,"fast in der Ebene\n→ nahe Rand",BLUE)]:
    ar=np.deg2rad(a); tip=np.array([np.cos(ar),np.sin(ar)])
    ax.annotate("",xy=tip,xytext=(0,0),arrowprops=dict(arrowstyle="->",color=c,lw=2))
    # projection from south pole (0,-1) through tip to equator (y=0)
    sp=np.array([0,-1.0]); t=(0-sp[1])/(tip[1]-sp[1]); P=sp+t*(tip-sp)
    ax.plot([sp[0],tip[0]],[sp[1],tip[1]],color=c,ls=":",lw=1.2)
    ax.plot(P[0],0,"o",color=c,ms=8)
    ax.text(tip[0]+0.03,tip[1]+0.03,lab,color=c,fontsize=8.5,va="bottom")
ax.text(0,-1.14,"Süd-Pol (Projektionszentrum)",ha="center",fontsize=8,color="0.4")
ax.text(0,0.06,"Normale ↑ (Blickrichtung)",ha="center",fontsize=8,color="0.35")
ax.set_title("Stereografische Projektion: Kristallrichtung (Pfeil) → Punkt auf der Scheibe",
             fontsize=10.5,color=NAVY)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.1)
fig.tight_layout(); fig.savefig("expl_C_stereo.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- D: schematic {100} pole figures: random vs sharp cube ----------
fig,axs=plt.subplots(1,2,figsize=(10,5))
rng=np.random.default_rng(7)
for ax,(title,mode) in zip(axs,[("regellose Textur","rand"),("scharfe ⟨100⟩-Textur","cube")]):
    t=np.linspace(0,2*np.pi,200); ax.plot(np.cos(t),np.sin(t),"k",lw=1.4); ax.plot(0,0,"r+",ms=11,mew=2)
    if mode=="rand":
        n=260; rr=np.sqrt(rng.random(n)); aa=rng.uniform(0,2*np.pi,n)
        ax.scatter(rr*np.cos(aa),rr*np.sin(aa),s=14,c=NAVY,alpha=0.5,edgecolors="none")
    else:
        # cube: <100> ∥ normal -> centre cluster; other two in-plane -> 4 rim clusters
        def blob(cx,cy,n=70,sd=0.06):
            ax.scatter(cx+rng.normal(0,sd,n),cy+rng.normal(0,sd,n),s=16,c=NAVY,alpha=0.55,edgecolors="none")
        blob(0,0)
        for (cx,cy) in [(0.92,0),(-0.92,0),(0,0.92),(0,-0.92)]: blob(cx,cy,45,0.05)
    ax.set_title(title,fontsize=11,color=NAVY); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.25,1.15)
    ax.text(0,-1.16,"Zentrum = ⟨100⟩ ∥ Normale",ha="center",fontsize=8,color="0.4")
fig.suptitle("{100}-Polfigur — schematisch: Häufung = Textur, gleichmäßige Streuung = regellos",fontsize=11.5)
fig.tight_layout(); fig.savefig("expl_D_polefig.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote expl_A_cube.png expl_B_grains.png expl_C_stereo.png expl_D_polefig.png")
