import React, { useEffect, useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { LayoutDashboard, UserCog, Settings, LogOut, Sparkles } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import "./Dashboard.css";
import "./Profile.css";

interface OutfitSlot {
  item_id: string;
  category: string;
  tags: string[];
  image_url: string;
  occasion?: string;
  season?: string;
  similarity_score?: number;
}

interface OutfitItem {
  _id: string;
  anchor_item_id?: string;
  outfit_id?: number;
  outfit_score?: number;
  occasion?: string;
  season?: string;
  top?: OutfitSlot | null;
  bottom?: OutfitSlot | null;
  shoes?: OutfitSlot | null;
  outerwear?: OutfitSlot | null;
  sunglasses?: OutfitSlot | null;
  created_at?: string | null;
}

export function OutfitRecommendations() {
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
      icon: <UserCog className="text-white h-5 w-5 flex-shrink-0" />,
    },
    {
      label: "Outfit Recommendations",
      href: "/dashboard/outfit-recommendations",
      icon: <Sparkles className="text-white h-5 w-5 flex-shrink-0" />,
    },
    {
      label: "Settings",
      href: "/dashboard/settings",
      icon: <Settings className="text-white h-5 w-5 flex-shrink-0" />,
    },
    {
      label: "Logout",
      href: "/",
      icon: <LogOut className="text-white h-5 w-5 flex-shrink-0" />,
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
                  <div
                    key={idx}
                    onClick={link.label === "Logout" ? handleLogout : undefined}
                  >
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
        <OutfitRecommendationsContent />
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

const OutfitRecommendationsContent = () => {
  const [items, setItems] = useState<OutfitItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOutfits = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("http://localhost:5000/recommended-outfits");
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(
            err.error || `Failed to load recommended outfits (${res.status})`
          );
        }
        const data = await res.json();
        setItems(data.items || []);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load recommended outfits"
        );
      } finally {
        setLoading(false);
      }
    };

    fetchOutfits();
  }, []);

  return (
    <div className="flex flex-1 flex-col">
      <div className="p-2 md:p-10 rounded-tl-2xl border-l border-white/10 bg-black flex flex-col gap-4 flex-1 w-full h-full">
        <div className="mb-2">
          <h1 className="text-3xl font-bold text-white mb-2">
            Outfit Recommendations
          </h1>
          <p className="text-white/60">
            Curated outfit looks generated for you.
          </p>
        </div>

        {loading && (
          <p className="wardrobe-status">Loading recommended outfits...</p>
        )}
        {error && (
          <p className="wardrobe-status wardrobe-error">{error}</p>
        )}
        {!loading && !error && items.length === 0 && (
          <p className="wardrobe-status">
            No outfit recommendations available yet.
          </p>
        )}

        {items.length > 0 && (
          <div className="wardrobe-grid">
            {items.map((item) => {
              const parts: Array<{
                label: string;
                slot?: OutfitSlot | null;
              }> = [
                { label: "Top", slot: item.top },
                { label: "Bottom", slot: item.bottom },
                { label: "Shoes", slot: item.shoes },
              ];

              return (
                <div key={item._id} className="wardrobe-card">
                  <div className="outfit-parts-row">
                    {parts.map(({ label, slot }) => (
                      <div key={label} className="outfit-part">
                        <div className="outfit-part-label">{label}</div>
                        <div className="wardrobe-image-wrapper">
                          {slot?.image_url ? (
                            <img
                              src={slot.image_url}
                              alt={slot.tags?.[0] || `${label} item`}
                              className="wardrobe-image"
                            />
                          ) : (
                            <div className="wardrobe-placeholder">
                              No {label.toLowerCase()}
                            </div>
                          )}
                        </div>
                        {slot?.tags && slot.tags.length > 0 && (
                          <div className="outfit-tags">
                            <span className="outfit-tag-main">
                              {slot.tags[0]}
                            </span>
                            {slot.tags.slice(1, 3).map((t, idx) => (
                              <span key={idx} className="wardrobe-chip">
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default OutfitRecommendations;

