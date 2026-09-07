export const Icon = ({ name, className = "" }: { name: string; className?: string }) => {
  const paths: Record<string, React.ReactNode> = {
    upload: <><path d="M12 16V4m0 0 4.5 4.5M12 4 7.5 8.5"/><path d="M5 14v5h14v-5"/></>,
    record: <><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="9"/></>,
    play: <path d="m9 7 8 5-8 5V7Z"/>,
    pause: <><path d="M9 7v10M15 7v10"/></>,
    close: <><path d="m7 7 10 10M17 7 7 17"/></>,
    arrow: <><path d="M5 12h13M14 8l4 4-4 4"/></>,
    replay: <><path d="M7 8H3V4"/><path d="M4 8a8 8 0 1 1-.4 7"/></>,
    check: <path d="m6 12 4 4 8-9"/>,
    shield: <path d="M12 3 5 6v5c0 4.5 2.8 8 7 10 4.2-2 7-5.5 7-10V6l-7-3Z"/>,
  };
  return (
    <svg 
      className={className} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="1.7" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
};
