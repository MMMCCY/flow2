"""Publication renderer for exact voxel surfaces. No statistical transformations."""
from pathlib import Path
import html
import math
import textwrap
import numpy as np
from ensemble_occurrence import exposed_faces


def render_panels(out,panels,shape,spacing=(1,1,1),title='',front=False,z_down=False,columns=4,presentation=False):
    from PIL import Image,ImageDraw,ImageFont
    cols=min(columns,len(panels)); rows=math.ceil(len(panels)/cols)
    width=1920 if presentation else 520*cols
    height=1080 if presentation else 470*rows+110
    w=width/cols; h=(height-110)/rows
    image=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(image)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def font(size):
        try: return ImageFont.truetype(fontpath,size)
        except OSError: return ImageFont.load_default()
    titlefont,font_regular,font_small=font(19),font(15),font(12)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>']
    def text(x,y,value,size=15,color='#303030'):
        draw.text((int(x),int(y)),value,font=titlefont if size==19 else font_regular if size==15 else font_small,fill=color)
        svg.append(f'<text x="{x:.1f}" y="{y+size:.1f}" font-family="DejaVu Sans, sans-serif" font-size="{size}" fill="{color}">{html.escape(value)}</text>')
    for k,line in enumerate(textwrap.wrap(title,max(90,int(width/11)))): text(20,12+23*k,line,19)
    text(20,68,'Gold: target label9   |   Red: independent observed target voxel overlay   |   Exact unsmoothed cell faces',12)
    text(20,88,'Orthographic front (along +Y)' if front else 'Orthographic isometric (from -X, -Y, +Z)',12)
    # orthogonal camera basis; camera sees the model from above in isometric view.
    sign=-1 if z_down else 1
    basis=np.array([[1,0,0],[0,0,-sign],[0,1,0]],float) if front else np.array([[1/math.sqrt(2),-1/math.sqrt(2),0],[-1/math.sqrt(6),-1/math.sqrt(6),-2*sign/math.sqrt(6)],[1/math.sqrt(3),1/math.sqrt(3),-sign/math.sqrt(3)]])
    spacing=np.asarray(spacing,float)
    corners=np.array([[x,y,z] for x in (0,shape[0]) for y in (0,shape[1]) for z in (0,shape[2])])*spacing
    projected=corners@basis.T; lo=projected[:,:2].min(0); hi=projected[:,:2].max(0)
    scale=min((w-88)/max(hi[0]-lo[0],1),(h-96)/max(hi[1]-lo[1],1))
    geometry_width,geometry_height=(hi-lo)*scale
    for index,(name,layers) in enumerate(panels):
        px=(index%cols)*w; py=110+(index//cols)*h
        for line_index,line in enumerate(textwrap.wrap(name,max(28,int(w/10)))):
            text(px+16,py+8+19*line_index,line)
        ox=px+(w-geometry_width)/2; oy=py+50+(h-96-geometry_height)/2
        def project(points): return (points@basis.T)[...,:2]*scale-lo*scale+np.array([ox,oy])
        def line(points,color='#a7b0b6',line_width=1):
            q=project(np.asarray(points)); draw.line([tuple(v) for v in q],fill=color,width=line_width)
            svg.append('<polyline points="'+' '.join(f'{v[0]:.2f},{v[1]:.2f}' for v in q)+f'" fill="none" stroke="{color}" stroke-width="{line_width}"/>')
        for a in range(8):
            for b in range(a+1,8):
                if np.count_nonzero(corners[a]!=corners[b])==1: line([corners[a],corners[b]],'#c4cacf')
        target_empty=not layers or not np.any(layers[0][0])
        for mask,color in layers:
            faces=exposed_faces(mask,spacing)
            if not len(faces): continue
            face_axis=np.argmin(np.ptp(faces,axis=1),axis=1)
            depth=(faces.mean(1)@basis.T)[:,2]
            order=np.argsort(-depth,kind='stable')
            colors=[]
            rgb=np.array([int(color[k:k+2],16) for k in (1,3,5)])
            for factor in (0.83,0.95,1.0):
                shade=np.clip(rgb*factor,0,255).astype(int); colors.append('#'+''.join(f'{v:02x}' for v in shade))
            polys=project(faces)
            for face_index in order:
                polygon=polys[face_index]; shade=colors[face_axis[face_index]]
                draw.polygon([tuple(v) for v in polygon],fill=shade)
                svg.append('<polygon points="'+' '.join(f'{v[0]:.2f},{v[1]:.2f}' for v in polygon)+f'" fill="{shade}"/>')
        if target_empty and ('C>' in name or 'Empty' in name):
            text(px+20,py+h-47,'Empty threshold region (red overlay independent)',12,'#687278')
        # Label physical axes at their genuine projected endpoints. All panels share fixed extent.
        origin=np.zeros(3)
        for axis,label in enumerate(('X','Y','Z')):
            endpoint=np.zeros(3); endpoint[axis]=shape[axis]*spacing[axis]
            q=project([origin,endpoint]); line([origin,endpoint],'#707a80')
            text(q[1,0]+3,q[1,1]+2,label,12)
        extent='/'.join(f'{shape[i]*spacing[i]/1000:g}' for i in range(3))
        text(px+16,py+h-22,f'XYZ extent {extent} km; cell 100/100/50 m' if np.allclose(spacing,[100,100,50]) else 'XYZ spacing '+str(spacing.tolist()),12,'#667077')
    svg.append('</svg>'); out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    image.save(out.with_suffix('.png')); out.with_suffix('.svg').write_text('\n'.join(svg))
