"use strict";
const D=JSON.parse(document.getElementById('analysis-data').textContent);
const byId=id=>document.getElementById(id);
const groupNames={full_worst:'Full: sai và tự tin nhất',partial_worst:'Partial: sai và tự tin nhất',regression:'Scratch đúng → Full sai',partial_recovers:'Full sai → Partial đúng',partial_regresses:'Full đúng → Partial sai',heart_to_lung:'Tim → Phổi trái (Full)',spleen_to_lung:'Lách → Phổi trái (Full)',random_errors:'Lỗi Full ngẫu nhiên theo lớp',correct_controls:'Ảnh Full đúng để đối chiếu',focus_controls:'Ảnh đúng của các nhóm lỗi nổi bật',kidney_to_spleen:'Thận trái → Lách (Full)',heart_comparison:'Đối chiếu Tim / Phổi trái',kidney_comparison:'Đối chiếu Thận trái / Lách'};
function option(select,value,label){const node=document.createElement('option');node.value=value;node.textContent=label;select.append(node);}
Object.keys(D.groups).forEach(key=>option(byId('case-group'),key,groupNames[key]||key));
D.labels_vi.forEach((name,i)=>option(byId('case-label'),i,name));
function renderCases(){
 const group=byId('case-group').value,label=byId('case-label').value,stage=byId('case-model').value;
 const indices=D.groups[group];const cases=indices.map(index=>D.cases.find(row=>row.index===index)).filter(row=>label==='all'||row.label===Number(label));
 byId('case-count').textContent=`${cases.length} ảnh được chọn · thang xám 0–255 · thứ tự theo quy tắc chọn của nhóm`;
 byId('gallery').replaceChildren();
 for(const row of cases){const value=row.models[stage],card=document.createElement('article');card.className='case';
 const img=document.createElement('img');img.src=row.image;img.alt=`Test ${row.index}, nhãn thật ${D.labels_vi[row.label]}`;card.append(img);
 const title=document.createElement('strong');title.textContent=`#${row.index} · Thật: ${D.labels_vi[row.label]}`;card.append(title);
 for(const key of ['scratch','linear','finetune','partial']){const m=row.models[key],line=document.createElement('div');line.className=m.prediction===row.label?'good':'bad';line.textContent=`${key}: ${D.labels_vi[m.prediction]} · ${(m.confidence*100).toFixed(1)}%`;card.append(line);}
 const loss=document.createElement('div');loss.textContent=`${stage}: P(nhãn thật)=${(value.true_probability*100).toFixed(3)}% · CE=${(-Math.log(Math.max(value.true_probability,1e-30))).toFixed(2)}`;card.append(loss);byId('gallery').append(card);}
}
['case-group','case-label','case-model'].forEach(id=>byId(id).addEventListener('change',renderCases));renderCases();
Object.keys(D.coordinates).forEach(stage=>option(byId('embedding-stage'),stage,D.names[stage]));byId('embedding-stage').value='mae';
const active=new Set(D.labels.map((_,i)=>i));let displayed=[],selected=-1;
D.labels_vi.forEach((name,i)=>{const button=document.createElement('button');button.setAttribute('aria-pressed','true');const dot=document.createElement('span');dot.className='dot';dot.style.background=D.colors[i];button.append(dot,document.createTextNode(name));button.addEventListener('click',()=>{if(active.has(i))active.delete(i);else active.add(i);button.setAttribute('aria-pressed',String(active.has(i)));drawEmbedding();});byId('embedding-legend').append(button);});
function drawEmbedding(){
 const canvas=byId('embedding-canvas'),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
 canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,rect.width,rect.height);
 const stage=byId('embedding-stage').value,method=byId('embedding-method').value,xy=D.coordinates[stage][method],prediction=D.test_predictions[stage];
 const wrongOnly=byId('embedding-filter').value==='wrong';let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
 for(const point of xy){xmin=Math.min(xmin,point[0]);xmax=Math.max(xmax,point[0]);ymin=Math.min(ymin,point[1]);ymax=Math.max(ymax,point[1]);}
 const left=56,top=22,width=rect.width-80,height=rect.height-70,dx=Math.max(xmax-xmin,1e-12),dy=Math.max(ymax-ymin,1e-12);
 displayed=[];ctx.strokeStyle='#b8c9d6';ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,top+height);ctx.lineTo(left+width,top+height);ctx.stroke();
 ctx.font='12px Arial';ctx.fillStyle='#53677a';for(let tick=0;tick<5;tick++){const f=tick/4;ctx.fillText((xmin+f*dx).toFixed(1),left+width*f-12,top+height+20);ctx.fillText((ymin+f*dy).toFixed(1),2,top+height*(1-f)+4);}
 for(let label=0;label<11;label++){if(!active.has(label))continue;ctx.fillStyle=D.colors[label];ctx.globalAlpha=.42;
 for(let i=0;i<xy.length;i++){if(D.test_labels[i]!==label||(wrongOnly&&(!prediction||prediction[i]===label)))continue;
 const x=left+(xy[i][0]-xmin)/dx*width,y=top+height-(xy[i][1]-ymin)/dy*height;ctx.fillRect(x-1.2,y-1.2,2.4,2.4);displayed.push({x,y,index:i});}}
 ctx.globalAlpha=1;const meta=D.projections[stage][method];byId('embedding-info').textContent=`${displayed.length.toLocaleString('vi-VN')} / 17.778 ảnh · ${method.toUpperCase()} giữ ${(meta.centered_variance_ratio.reduce((a,b)=>a+b,0)*100).toFixed(2)}% phương sai 384D (đã trừ mean)${wrongOnly&&!prediction?' · Encoder này chưa có classifier; không định nghĩa ảnh sai.':''}`;
}
['embedding-stage','embedding-method','embedding-filter'].forEach(id=>byId(id).addEventListener('change',drawEmbedding));
byId('embedding-canvas').addEventListener('pointermove',event=>{const rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=144,chosen=null;
 for(const point of displayed){const distance=(point.x-x)**2+(point.y-y)**2;if(distance<best){best=distance;chosen=point;}}
 const tip=byId('point-tooltip');if(!chosen){tip.hidden=true;return;}selected=chosen.index;const truth=D.test_labels[selected],stage=byId('embedding-stage').value,pred=D.test_predictions[stage];
 tip.hidden=false;tip.style.left=`${Math.min(x+10,Math.max(0,rect.width-230))}px`;tip.style.top=`${Math.max(0,y-45)}px`;tip.textContent=`#${selected} · ${D.labels_vi[truth]}${pred?' → '+D.labels_vi[pred[selected]]:''}`;
 byId('point-detail').textContent=`Test #${selected} · Thật: ${D.labels_vi[truth]} · Full: ${D.labels_vi[D.test_predictions.finetune[selected]]} · Partial: ${D.labels_vi[D.test_predictions.partial[selected]]}. Điểm gần nhau trong 2D có thể vẫn cách xa trong 384D.`;
});
byId('embedding-canvas').addEventListener('pointerleave',()=>{byId('point-tooltip').hidden=true;});
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawEmbedding,100);});drawEmbedding();
