import { useEffect, useState } from 'react';

const WARMUP_TOTAL_MS = 30_000;

interface WarmUpBannerProps {
  visible: boolean;
}

export function WarmUpBanner({ visible }: WarmUpBannerProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!visible) {
      setElapsed(0);
      return;
    }
    const start = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.min(Date.now() - start, WARMUP_TOTAL_MS));
    }, 250);
    return () => clearInterval(id);
  }, [visible]);

  if (!visible) return null;

  const progress = Math.min((elapsed / WARMUP_TOTAL_MS) * 100, 95);
  const remaining = Math.max(0, Math.round((WARMUP_TOTAL_MS - elapsed) / 1000));

  return (
    <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-4 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-medium text-yellow-400">⚡ GPU warming up</span>
        <span className="text-yellow-400/70">~{remaining}s remaining</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-yellow-500/20">
        <div
          className="h-full rounded-full bg-yellow-400 transition-all duration-[250ms]"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="mt-2 text-yellow-400/60">
        This only happens after a period of inactivity.
      </p>
    </div>
  );
}
