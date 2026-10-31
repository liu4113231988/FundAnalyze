/* Progressive field enhancement: native values/validation remain the contract. */
'use strict';
(() => {
  if (!HTMLElement.prototype.showPopover) return;
  const enhanced = new WeakMap();
  let active = null, serial = 0;
  const make = (tag, cls, text) => {const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;};
  const dateAt = (y,m,d) => {const value=new Date(0);value.setFullYear(y,m,d);value.setHours(0,0,0,0);return value;};
  const iso = d => `${String(d.getFullYear()).padStart(4,'0')}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
  function parse(value) {const match=/^(\d{4})-(\d{2})-(\d{2})$/.exec(value);if(!match)return null;const d=dateAt(+match[1],+match[2]-1,+match[3]);return iso(d)===value&&d.getFullYear()>0?d:null;}
  function labelFor(native) {
    return native.getAttribute('aria-label') || [...native.labels||[]].map(l=>[...l.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('').trim()).filter(Boolean).join(' ') || native.title || ({btStart:'回测开始日期',btEnd:'回测结束日期',btModel:'回测模型',rebalSel:'重估点',windowSel:'分析窗口'}[native.id]) || '选择';
  }
  function close(restore=false) {if(!active)return;const previous=active;active=null;previous.anchor.setAttribute('aria-expanded','false');if(previous.pop.matches(':popover-open'))previous.pop.hidePopover();previous.pop.remove();if(restore)previous.anchor.focus();}
  function position() {
    if(!active)return;
    const {anchor,pop}=active, rect=(anchor.closest('.control-date')||anchor).getBoundingClientRect(), margin=12;
    const width=Math.min(pop.classList.contains('calendar-popover')?352:Math.max(rect.width,220),innerWidth-margin*2);
    pop.style.width=width+'px';pop.style.maxHeight=Math.max(150,innerHeight-margin*2)+'px';
    const height=pop.getBoundingClientRect().height;
    const below=innerHeight-rect.bottom-margin,above=rect.top-margin;
    const top=below>=height?rect.bottom+8:above>=height?rect.top-height-8:Math.max(margin,Math.min(rect.bottom+8,innerHeight-height-margin));
    pop.style.left=Math.max(margin,Math.min(rect.left,innerWidth-width-margin))+'px';pop.style.top=top+'px';
  }
  function open(anchor,pop) {
    close();pop.setAttribute('popover','manual');(anchor.closest('dialog')||document.body).append(pop);
    anchor.setAttribute('aria-expanded','true');active={anchor,pop};pop.showPopover();position();
    pop.addEventListener('keydown',event=>{
      if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close(true);}
      else if(event.key==='Tab'&&active&&pop.classList.contains('calendar-popover')){
        const targets=[...pop.querySelectorAll('button:not(:disabled),input:not(:disabled)')].filter(e=>e.tabIndex>=0);
        if((event.shiftKey&&event.target===targets[0])||(!event.shiftKey&&event.target===targets.at(-1)))close(true);
      }
    });
  }
  function dispatch(native) {native.dispatchEvent(new Event('input',{bubbles:true}));native.dispatchEvent(new Event('change',{bubbles:true}));}
  function watchValue(native,property,sync) {
    const descriptor=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(native),property);
    if(!descriptor?.set)return;
    Object.defineProperty(native,property,{configurable:true,get(){return descriptor.get.call(this);},set(value){descriptor.set.call(this,value);sync();}});
  }
  function base(native,cls) {
    const wrapper=make('span','control-field '+cls);native.before(wrapper);wrapper.append(native);native.classList.add('control-native');native.tabIndex=-1;native.setAttribute('aria-hidden','true');
    return wrapper;
  }
  function select(native) {
    if(native.multiple||native.size>1)return;
    const wrapper=base(native,'control-select'), button=make('button','control-trigger');
    button.type='button';button.setAttribute('role','combobox');button.setAttribute('aria-haspopup','listbox');button.setAttribute('aria-expanded','false');
    const value=make('span','control-value'),chevron=make('span','control-chevron','⌄');chevron.setAttribute('aria-hidden','true');button.append(value,chevron);wrapper.append(button);
    let list, optionButtons=[], text='', typedAt=0;
    function sync() {value.textContent=native.selectedOptions[0]?.label||'请选择';button.disabled=native.disabled;button.setAttribute('aria-label',labelFor(native)+'：'+value.textContent);button.title=value.textContent;wrapper.hidden=native.hidden;wrapper.classList.toggle('is-disabled',native.disabled);if(native.validity.valid)button.removeAttribute('aria-invalid');if(active?.anchor===button)close();}
    function commit(index) {const option=native.options[index];if(!option||option.disabled||option.parentElement.disabled)return;native.selectedIndex=index;dispatch(native);close();button.focus();}
    function show(focusIndex=native.selectedIndex) {
      if(native.disabled)return;
      list=make('div','control-popover select-popover');list.id='control-options-'+(++serial);list.setAttribute('role','listbox');list.setAttribute('aria-label',labelFor(native));button.setAttribute('aria-controls',list.id);
      optionButtons=[];
      [...native.children].forEach(group=>{
        if(group.tagName==='OPTGROUP')list.append(make('p','select-group',group.label));
        const options=group.tagName==='OPTGROUP'?[...group.children]:[group];
        options.forEach(option=>{if(option.hidden)return;const index=[...native.options].indexOf(option);if(index<0)return;const item=make('button','select-option');item.type='button';item.tabIndex=-1;item.dataset.index=index;item.setAttribute('role','option');item.setAttribute('aria-selected',String(option.selected));item.disabled=option.disabled||!!option.parentElement.disabled;
          item.append(make('span','',option.label),make('span','select-check',option.selected?'✓':''));item.onclick=()=>commit(index);list.append(item);optionButtons.push(item);});
      });
      if(!optionButtons.length)list.append(make('p','control-empty','暂无可选项'));
      list.addEventListener('keydown',key);
      open(button,list);const focused=optionButtons.find(b=>+b.dataset.index===focusIndex&&!b.disabled)||optionButtons.find(b=>!b.disabled);if(focused){focused.focus();focused.scrollIntoView({block:'nearest'});}
    }
    function key(event) {
      const available=optionButtons.filter(b=>!b.disabled);let current=available.indexOf(document.activeElement);
      if((event.key==='Enter'||event.key===' ')&&active?.anchor===button&&current>=0){event.preventDefault();commit(+available[current].dataset.index);}
      else if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)) {
        event.preventDefault();if(active?.anchor!==button){show(event.key==='ArrowUp'?native.options.length-1:native.selectedIndex);return;}
        current=event.key==='Home'?0:event.key==='End'?available.length-1:Math.max(0,Math.min(available.length-1,current+(event.key==='ArrowDown'?1:-1)));available[current]?.focus();
      } else if(event.key==='Tab'&&active?.anchor===button){close();button.focus();}
      else if(event.key==='Escape'&&active?.anchor===button){event.preventDefault();event.stopPropagation();close(true);}
      else if(event.key.length===1&&!event.ctrlKey&&!event.metaKey&&event.key!==' ') {
        event.preventDefault();text=Date.now()-typedAt>700?event.key:text+event.key;typedAt=Date.now();if(active?.anchor!==button)show();const match=optionButtons.find(b=>!b.disabled&&b.textContent.toLowerCase().startsWith(text.toLowerCase()));match?.focus();
      }
    }
    button.onclick=()=>active?.anchor===button?close():show();button.addEventListener('keydown',key);
    native.addEventListener('change',sync);native.addEventListener('invalid',event=>{event.preventDefault();button.focus();button.setAttribute('aria-invalid','true');});
    watchValue(native,'value',sync);watchValue(native,'selectedIndex',sync);
    enhanced.set(native,{sync,wrapper,focus:()=>button.focus()});sync();
  }
  function calendar(native) {
    const wrapper=base(native,'control-date'), input=make('input','control-date-text'), button=make('button','control-calendar-button','');
    input.type='text';input.placeholder='YYYY-MM-DD';input.inputMode='numeric';input.maxLength=10;input.autocomplete='off';input.spellcheck=false;
    button.type='button';button.innerHTML='<svg viewBox="0 0 24 24" width="17" height="17" fill="none" aria-hidden="true"><rect x="4" y="5" width="16" height="16" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M8 3v4m8-4v4M4 10h16m-11 4h2m2 0h2m-6 3h2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
    button.setAttribute('aria-haspopup','dialog');button.setAttribute('aria-expanded','false');wrapper.append(input,button);
    let view,mode='days',pop,focused;
    const allowed=d=>d.getFullYear()>=1&&d.getFullYear()<=9999&&(!native.min||iso(d)>=native.min)&&(!native.max||iso(d)<=native.max);
    function sync() {input.value=native.value;input.required=native.required;input.disabled=button.disabled=native.disabled;input.setAttribute('aria-label',labelFor(native));button.setAttribute('aria-label','打开'+labelFor(native)+'日历');wrapper.hidden=native.hidden;wrapper.classList.toggle('is-disabled',native.disabled);input.setCustomValidity('');input.removeAttribute('aria-invalid');}
    function commit(d) {if(d&&!allowed(d))return;native.value=d?iso(d):'';dispatch(native);close();input.focus();}
    function render(focusDay=false) {
      const previousDirection=document.activeElement?.dataset.direction;
      pop.replaceChildren();const title=make('div','calendar-heading');
      const prev=make('button','calendar-nav','‹'),next=make('button','calendar-nav','›');prev.type=next.type='button';prev.setAttribute('aria-label','上一'+(mode==='years'?'组年份':mode==='months'?'年':'月'));next.setAttribute('aria-label','下一'+(mode==='years'?'组年份':mode==='months'?'年':'月'));
      const year=make('button','calendar-title',view.getFullYear()+'年'),month=make('button','calendar-title',(view.getMonth()+1)+'月');year.type=month.type='button';year.setAttribute('aria-label','选择年份');month.setAttribute('aria-label','选择月份');
      year.onclick=()=>{mode=mode==='years'?'days':'years';render();};month.onclick=()=>{mode=mode==='months'?'days':'months';render();};
      prev.onclick=()=>move(-1);next.onclick=()=>move(1);title.append(prev,year,month,next);pop.append(title);
      prev.dataset.direction='previous';next.dataset.direction='next';
      const body=make('div',mode==='days'?'calendar-grid':'calendar-period-grid');
      if(mode==='days') {
        ['一','二','三','四','五','六','日'].forEach(day=>body.append(make('span','calendar-weekday',day)));
        const first=dateAt(view.getFullYear(),view.getMonth(),1),offset=(first.getDay()+6)%7;
        for(let i=0;i<42;i++) {
          const day=dateAt(view.getFullYear(),view.getMonth(),1-offset+i),date=iso(day),cell=make('button','calendar-day',String(day.getDate()));cell.type='button';cell.dataset.date=date;cell.setAttribute('aria-label',date);cell.disabled=!allowed(day);
          cell.classList.toggle('other-month',day.getMonth()!==view.getMonth());cell.classList.toggle('is-today',date===iso(new Date()));cell.classList.toggle('is-selected',date===native.value);if(date===native.value)cell.setAttribute('aria-pressed','true');if(date===iso(new Date()))cell.setAttribute('aria-current','date');
          cell.tabIndex=date===iso(focused)?0:-1;cell.onclick=()=>commit(day);cell.onkeydown=event=>{
            const delta={ArrowLeft:-1,ArrowRight:1,ArrowUp:-7,ArrowDown:7}[event.key];
            if(delta){event.preventDefault();const target=dateAt(day.getFullYear(),day.getMonth(),day.getDate()+delta);if(allowed(target)){focused=target;view=dateAt(target.getFullYear(),target.getMonth(),1);render(true);}}
            else if(['PageUp','PageDown'].includes(event.key)){event.preventDefault();move(event.key==='PageUp'?-1:1,true);}
          };body.append(cell);
        }
      } else {
        const start=mode==='years'?Math.max(1,Math.floor((view.getFullYear()-1)/12)*12+1):1;
        for(let i=0;i<12;i++) {const value=start+i,cell=make('button','calendar-period',value+(mode==='years'?'年':'月'));cell.type='button';cell.disabled=mode==='years'&&value>9999;cell.classList.toggle('is-selected',mode==='years'?value===view.getFullYear():value===view.getMonth()+1);cell.onclick=()=>{view=dateAt(mode==='years'?value:view.getFullYear(),mode==='months'?value-1:view.getMonth(),1);focused=view;mode=mode==='years'?'months':'days';render(mode==='days');};body.append(cell);}
      }
      pop.append(body);const footer=make('div','calendar-footer'),hint=make('span','',native.required?'必填日期':'可选日期');footer.append(hint);
      if(!native.required){const clear=make('button','','清空');clear.type='button';clear.onclick=()=>commit(null);footer.append(clear);}
      const today=make('button','','今天');today.type='button';today.disabled=!allowed(new Date());today.onclick=()=>commit(new Date());footer.append(today);pop.append(footer);position();
      if(mode!=='days')body.querySelector('.is-selected')?.focus();
      else if(focusDay)pop.querySelector(`[data-date="${iso(focused)}"]`)?.focus();
      else if(previousDirection)pop.querySelector(`[data-direction="${previousDirection}"]`)?.focus();
    }
    function move(direction,focusDay=false) {const candidate=dateAt(view.getFullYear()+(mode==='years'?12*direction:mode==='months'?direction:0),view.getMonth()+(mode==='days'?direction:0),1);if(candidate.getFullYear()<1||candidate.getFullYear()>9999)return;view=candidate;focused=view;render(focusDay);}
    function show() {if(native.disabled)return;if(active?.anchor===button){close();return;}focused=parse(native.value)||new Date();view=dateAt(focused.getFullYear(),focused.getMonth(),1);mode='days';pop=make('div','control-popover calendar-popover');pop.id='control-calendar-'+(++serial);pop.setAttribute('role','dialog');pop.setAttribute('aria-label',labelFor(native)+'选择日历');button.setAttribute('aria-controls',pop.id);open(button,pop);render(true);}
    function manual() {const value=input.value.trim().replaceAll('/','-').replace(/^(\d{4})(\d{2})(\d{2})$/,'$1-$2-$3');const date=parse(value);if(!value){native.value='';input.setCustomValidity(native.required?'请选择日期':'');}else if(!date||!allowed(date)){native.value='';input.setCustomValidity('请输入有效的 YYYY-MM-DD 日期，并遵守日期范围');input.value=value;input.setAttribute('aria-invalid','true');return;}else native.value=value;dispatch(native);}
    input.onchange=manual;input.addEventListener('input',()=>{input.setCustomValidity('');input.removeAttribute('aria-invalid');});input.addEventListener('keydown',event=>{if(event.key==='ArrowDown'){event.preventDefault();show();}});button.onclick=show;
    native.addEventListener('change',sync);native.addEventListener('invalid',event=>{event.preventDefault();input.focus();input.setAttribute('aria-invalid','true');input.setCustomValidity(native.validationMessage);input.reportValidity();});
    watchValue(native,'value',sync);enhanced.set(native,{sync,wrapper,focus:()=>input.focus()});sync();
  }
  function scan(root=document) {root.querySelectorAll('select,input[type="date"]').forEach(native=>{if(!enhanced.has(native)){native.tagName==='SELECT'?select(native):calendar(native);}});}
  const observer=new MutationObserver(changes=>{
    let added=false;const updates=new Set();
    for(const change of changes){if(change.type==='childList'&&change.addedNodes.length)added=true;const target=change.target.closest?.('select,input[type="date"]');if(target&&enhanced.has(target))updates.add(target);}
    if(added)scan();updates.forEach(native=>enhanced.get(native).sync());
  });
  scan();observer.observe(document.body,{subtree:true,childList:true,attributes:true,attributeFilter:['disabled','selected','hidden','min','max','required','aria-label','title']});
  document.addEventListener('pointerdown',event=>{if(active&&!active.pop.contains(event.target)&&!active.anchor.closest('.control-field').contains(event.target))close();},true);
  document.addEventListener('focusin',event=>{if(active&&!active.pop.contains(event.target)&&!active.anchor.closest('.control-field').contains(event.target))close();});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&active){event.preventDefault();event.stopPropagation();close(true);}},true);
  document.addEventListener('click',event=>{const label=event.target.closest('label');if(!label||event.target.closest('button,input,.control-popover'))return;const native=label.control;if(native&&enhanced.has(native)){event.preventDefault();enhanced.get(native).focus();}});
  document.addEventListener('reset',()=>setTimeout(()=>document.querySelectorAll('select,input[type="date"]').forEach(n=>enhanced.get(n)?.sync())));
  window.addEventListener('resize',position);window.addEventListener('scroll',event=>{if(active&&!active.pop.contains(event.target))position();},true);
})();
