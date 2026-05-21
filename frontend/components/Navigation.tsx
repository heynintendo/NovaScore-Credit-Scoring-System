"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, useScroll, useMotionValueEvent } from "motion/react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Index" },
  { href: "/score", label: "Calculator" },
  { href: "/about", label: "Methodology" },
];

export function Navigation() {
  const pathname = usePathname();
  const { scrollY } = useScroll();
  const [scrolled, setScrolled] = useState(false);

  useMotionValueEvent(scrollY, "change", (v) => {
    setScrolled(v > 24);
  });

  return (
    <motion.nav
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-[background,backdrop-filter,border-color,padding] duration-500",
        scrolled
          ? "border-b border-white/[0.05] py-3 backdrop-blur-xl bg-[rgba(10,22,40,0.78)]"
          : "border-b border-transparent py-5 bg-transparent",
      )}
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 lg:px-10">
        <Link href="/" className="group flex items-baseline gap-3" aria-label="NovaScore home">
          {/* Custom monogram — small Fraunces N with a thin gold underline. */}
          <span className="relative font-display text-2xl font-semibold tracking-tight text-cream">
            Nova
            <span className="text-gold">Score</span>
            <span className="absolute -bottom-1 left-0 h-px w-0 bg-gold transition-all duration-500 group-hover:w-full" />
          </span>
        </Link>

        <ul className="flex items-center gap-1 sm:gap-2">
          {LINKS.map((link) => {
            const active =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);
            return (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className={cn(
                    "relative rounded-full px-3 py-2 text-sm transition-colors sm:px-4",
                    active ? "text-cream" : "text-cream-muted hover:text-cream",
                  )}
                >
                  {link.label}
                  {active && (
                    <motion.span
                      layoutId="nav-pill"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                      className="absolute inset-0 -z-10 rounded-full bg-white/[0.04] ring-1 ring-white/[0.06]"
                    />
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
    </motion.nav>
  );
}
