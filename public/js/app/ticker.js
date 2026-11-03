// Timers that keep their rate in a hidden tab. A hidden tab's own timers
// are throttled (Chrome: at most one wake-up a second, after five minutes
// one a minute) and its animation frames stop; a dedicated worker's timers
// are not, and the messages it posts are handled at once. So what has to go
// on while the tab is in the background (behaviour trees, ROS, the shared
// world's pings, the world itself: main.js background) runs on a worker's
// interval: every(ms, fn) calls fn about every ms, and the function it
// returns stops it. A call is not sent again before fn has returned, so a
// slow fn is not queued up. Without workers, setInterval.

const WORKER = `
const timers = new Map(), out = new Set();
onmessage = ({ data: [id, ms] }) => {
  if (ms === 'done') { out.delete(id); return; }
  clearInterval(timers.get(id));
  timers.delete(id);
  out.delete(id);
  if (ms) timers.set(id, setInterval(() => { if (!out.has(id)) { out.add(id); postMessage(id); } }, ms));
};`;

let worker = null;  // null: not made yet; false: none to be had
const calls = new Map();  // id -> fn
let next = 1;

function start() {
  try {
    worker = new Worker(URL.createObjectURL(new Blob([WORKER], { type: 'text/javascript' })));
    worker.onmessage = ({ data: id }) => {
      const fn = calls.get(id);
      if (!fn) return;
      try {
        fn();
      } finally {
        if (calls.has(id)) worker.postMessage([id, 'done']);
      }
    };
  } catch {
    worker = false;
  }
}

export function every(ms, fn) {
  if (worker === null) start();
  if (!worker) {
    const timer = setInterval(fn, ms);
    return () => clearInterval(timer);
  }
  const id = next++;
  calls.set(id, fn);
  worker.postMessage([id, ms]);
  return () => {
    if (calls.delete(id)) worker.postMessage([id, 0]);
  };
}
