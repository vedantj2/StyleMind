import React, { useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { LayoutDashboard, UserCog, Settings, LogOut, Home } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import "./Dashboard.css";

export function Dashboard() {
  const navigate = useNavigate();
  const links = [
    {
      label: "Dashboard",
      href: "/dashboard",
      icon: (
        <LayoutDashboard className="text-white h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Profile",
      href: "/dashboard/profile",
      icon: (
        <UserCog className="text-white h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Settings",
      href: "/dashboard/settings",
      icon: (
        <Settings className="text-white h-5 w-5 flex-shrink-0" />
      ),
    },
    {
      label: "Logout",
      href: "/",
      icon: (
        <LogOut className="text-white h-5 w-5 flex-shrink-0" />
      ),
    },
  ];
  const [open, setOpen] = useState(false);

  const handleLogout = () => {
    localStorage.removeItem("rememberMe");
    navigate("/");
  };

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        <Sidebar open={open} setOpen={setOpen}>
          <SidebarBody className="justify-between gap-10">
            <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
              {open ? <Logo /> : <LogoIcon />}
              <div className="mt-8 flex flex-col gap-2">
                {links.map((link, idx) => (
                  <div key={idx} onClick={link.label === "Logout" ? handleLogout : undefined}>
                    <SidebarLink link={link} />
                  </div>
                ))}
              </div>
            </div>
            <div>
              <SidebarLink
                link={{
                  label: "User",
                  href: "/dashboard/profile",
                  icon: (
                    <div className="h-7 w-7 flex-shrink-0 rounded-full bg-white/20 flex items-center justify-center text-white text-xs">
                      U
                    </div>
                  ),
                }}
              />
            </div>
          </SidebarBody>
        </Sidebar>
        <DashboardContent />
      </div>
    </div>
  );
}

export const Logo = () => {
  return (
    <Link
      to="/"
      className="font-normal flex space-x-2 items-center text-sm text-white py-1 relative z-20"
    >
      <div className="h-5 w-6 bg-white rounded-br-lg rounded-tr-sm rounded-tl-lg rounded-bl-sm flex-shrink-0" />
      <motion.span
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="font-medium text-white whitespace-pre"
      >
        StyleMind
      </motion.span>
    </Link>
  );
};

export const LogoIcon = () => {
  return (
    <Link
      to="/"
      className="font-normal flex space-x-2 items-center text-sm text-white py-1 relative z-20"
    >
      <div className="h-5 w-6 bg-white rounded-br-lg rounded-tr-sm rounded-tl-lg rounded-bl-sm flex-shrink-0" />
    </Link>
  );
};

// Dashboard content component
const DashboardContent = () => {
  const stats = [
    {
      title: "Your Wardrobe",
      value: "247",
      subtitle: "Items cataloged",
      icon: "👔",
      trend: "+12 this month"
    },
    {
      title: "Your Requests",
      value: "8",
      subtitle: "Pending requests",
      icon: "📋",
      trend: "3 new today"
    },
    {
      title: "Following",
      value: "156",
      subtitle: "Fashion enthusiasts",
      icon: "👥",
      trend: "+5 this week"
    },
    {
      title: "Followers",
      value: "892",
      subtitle: "People following you",
      icon: "⭐",
      trend: "+23 this week"
    }
  ];

  return (
    <div className="flex flex-1 flex-col">
      <div className="p-2 md:p-10 rounded-tl-2xl border-l border-white/10 bg-black flex flex-col gap-2 flex-1 w-full h-full">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
          <p className="text-white/60">Welcome back! Here's your overview.</p>
        </div>
        <div className="flex gap-4 flex-wrap">
          {stats.map((stat, i) => (
            <div
              key={i}
              className="h-40 w-full md:w-[calc(50%-0.5rem)] lg:w-[calc(25%-0.75rem)] rounded-lg bg-[#1a1a1a] border border-white/10 p-6 hover:border-white/20 transition-colors flex flex-col items-center justify-center text-center"
            >
              <div className="flex flex-col items-center mb-4">
                <span className="text-2xl mb-2">{stat.icon}</span>
                <p className="text-white/60 text-sm mb-1">{stat.title}</p>
                <p className="text-3xl font-bold text-white">{stat.value}</p>
              </div>
              <p className="text-white/40 text-xs mb-1">{stat.subtitle}</p>
              <p className="text-white/60 text-xs">{stat.trend}</p>
            </div>
          ))}
        </div>
        <div className="flex gap-4 flex-1 mt-4">
          <div className="h-full w-full rounded-lg bg-[#1a1a1a] border border-white/10 p-6">
            <h3 className="text-white text-lg font-semibold mb-4">Recent Activity</h3>
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-black/50">
                  <div className="h-10 w-10 rounded-full bg-white/10 flex items-center justify-center">
                    <span className="text-white text-sm">📸</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-white text-sm">New item added to wardrobe</p>
                    <p className="text-white/40 text-xs">2 hours ago</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="h-full w-full rounded-lg bg-[#1a1a1a] border border-white/10 p-6">
            <h3 className="text-white text-lg font-semibold mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <button className="w-full p-3 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm text-left transition-colors">
                ➕ Add new item to wardrobe
              </button>
              <button className="w-full p-3 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm text-left transition-colors">
                🔍 Browse marketplace
              </button>
              <button className="w-full p-3 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm text-left transition-colors">
                👔 Get outfit recommendation
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

