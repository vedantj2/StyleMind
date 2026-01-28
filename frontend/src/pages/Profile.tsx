import React, { useEffect, useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { LayoutDashboard, UserCog, Settings, LogOut, Camera, Edit2, Save, X, Sparkles } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import "./Profile.css";

export function Profile() {
  const navigate = useNavigate();
  const [isEditing, setIsEditing] = useState(false);
  const [profileData, setProfileData] = useState({
    name: "John Doe",
    username: "@johndoe",
    bio: "Fashion enthusiast and style curator. Love exploring new trends and building sustainable wardrobes.",
    email: "john.doe@example.com",
    location: "New York, USA",
    website: "www.johndoe.com",
    joinDate: "January 2024"
  });

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
      label: "Outfit Recommendations",
      href: "/dashboard/outfit-recommendations",
      icon: (
        <Sparkles className="text-white h-5 w-5 flex-shrink-0" />
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
    localStorage.removeItem("isLoggedIn");
    navigate("/");
  };

  const handleSave = () => {
    setIsEditing(false);
    // Here you would typically save to backend
    console.log("Profile saved:", profileData);
  };

  const handleCancel = () => {
    setIsEditing(false);
    // Reset to original values if needed
  };

  return (
    <div className="profile-page">
      <div className="profile-container">
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
        <ProfileContent 
          profileData={profileData}
          setProfileData={setProfileData}
          isEditing={isEditing}
          setIsEditing={setIsEditing}
          onSave={handleSave}
          onCancel={handleCancel}
        />
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

interface ProfileContentProps {
  profileData: {
    name: string;
    username: string;
    bio: string;
    email: string;
    location: string;
    website: string;
    joinDate: string;
  };
  setProfileData: React.Dispatch<React.SetStateAction<{
    name: string;
    username: string;
    bio: string;
    email: string;
    location: string;
    website: string;
    joinDate: string;
  }>>;
  isEditing: boolean;
  setIsEditing: React.Dispatch<React.SetStateAction<boolean>>;
  onSave: () => void;
  onCancel: () => void;
}

const ProfileContent = ({ 
  profileData, 
  setProfileData, 
  isEditing, 
  setIsEditing, 
  onSave, 
  onCancel 
}: ProfileContentProps) => {
  const [profileImage, setProfileImage] = useState("https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=400&fit=crop");
  const [wardrobeItems, setWardrobeItems] = useState<Array<{
    _id: string;
    url: string;
    created_at?: string | null;
    tags?: {
      garment_type?: string;
      primary_color?: string;
      season?: string;
    };
  }>>([]);
  const [wardrobeLoading, setWardrobeLoading] = useState(false);
  const [wardrobeError, setWardrobeError] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");
  const [typeFilter, setTypeFilter] = useState<string>("all");

  useEffect(() => {
    const fetchWardrobeItems = async () => {
      setWardrobeLoading(true);
      setWardrobeError(null);
      try {
        const res = await fetch("http://localhost:5000/wardrobe-items");
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error || `Failed to load wardrobe items (${res.status})`);
        }
        const data = await res.json();
        setWardrobeItems(data.items || []);
      } catch (err) {
        setWardrobeError(err instanceof Error ? err.message : "Failed to load wardrobe items");
      } finally {
        setWardrobeLoading(false);
      }
    };

    fetchWardrobeItems();
  }, []);

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="p-2 md:p-10 bg-black flex flex-col gap-6 flex-1 w-full">
        {/* Profile Header */}
        <div className="profile-header">
          <div className="profile-cover">
            <img 
              src="https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=1600&h=400&fit=crop" 
              alt="Cover" 
              className="profile-cover-image"
            />
            {isEditing && (
              <button className="profile-cover-edit">
                <Camera className="w-4 h-4 mr-2" />
                Edit Cover Photo
              </button>
            )}
          </div>
          <div className="profile-info-section">
            <div className="profile-image-container">
              <img 
                src={profileImage} 
                alt="Profile" 
                className="profile-image"
              />
              {isEditing && (
                <button className="profile-image-edit">
                  <Camera className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="profile-details">
              {!isEditing ? (
                <>
                  <div className="flex items-center gap-4 mb-2 flex-wrap">
                    <h1 className="profile-name">{profileData.name}</h1>
                    <button 
                      onClick={() => setIsEditing(true)}
                      className="edit-button"
                    >
                      <Edit2 className="w-4 h-4 mr-2" />
                      Edit Profile
                    </button>
                  </div>
                  <p className="profile-username">{profileData.username}</p>
                  <p className="profile-bio">{profileData.bio}</p>
                  <div className="profile-meta">
                    <span>{profileData.location}</span>
                    <span>•</span>
                    <a href={`https://${profileData.website}`} className="profile-link">{profileData.website}</a>
                    <span>•</span>
                    <span>Joined {profileData.joinDate}</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-4 mb-4">
                    <h1 className="profile-name">Edit Profile</h1>
                    <div className="flex gap-2">
                      <button 
                        onClick={onSave}
                        className="save-button"
                      >
                        <Save className="w-4 h-4 mr-2" />
                        Save
                      </button>
                      <button 
                        onClick={onCancel}
                        className="cancel-button"
                      >
                        <X className="w-4 h-4 mr-2" />
                        Cancel
                      </button>
                    </div>
                  </div>
                  <div className="profile-edit-form">
                    <div className="form-group">
                      <label>Name</label>
                      <input
                        type="text"
                        value={profileData.name}
                        onChange={(e) => setProfileData({...profileData, name: e.target.value})}
                        className="form-input"
                      />
                    </div>
                    <div className="form-group">
                      <label>Username</label>
                      <input
                        type="text"
                        value={profileData.username}
                        onChange={(e) => setProfileData({...profileData, username: e.target.value})}
                        className="form-input"
                      />
                    </div>
                    <div className="form-group">
                      <label>Bio</label>
                      <textarea
                        value={profileData.bio}
                        onChange={(e) => setProfileData({...profileData, bio: e.target.value})}
                        className="form-textarea"
                        rows={4}
                      />
                    </div>
                    <div className="form-group">
                      <label>Email</label>
                      <input
                        type="email"
                        value={profileData.email}
                        onChange={(e) => setProfileData({...profileData, email: e.target.value})}
                        className="form-input"
                      />
                    </div>
                    <div className="form-group">
                      <label>Location</label>
                      <input
                        type="text"
                        value={profileData.location}
                        onChange={(e) => setProfileData({...profileData, location: e.target.value})}
                        className="form-input"
                      />
                    </div>
                    <div className="form-group">
                      <label>Website</label>
                      <input
                        type="text"
                        value={profileData.website}
                        onChange={(e) => setProfileData({...profileData, website: e.target.value})}
                        className="form-input"
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div className="profile-stats">
          <div className="stat-item">
            <div className="stat-value">247</div>
            <div className="stat-label">Wardrobe</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">156</div>
            <div className="stat-label">Following</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">892</div>
            <div className="stat-label">Followers</div>
          </div>
        </div>

        {/* Wardrobe grid below stats */}
        <div className="wardrobe-section">
          <h2 className="wardrobe-title">Wardrobe Items</h2>
          {wardrobeLoading && (
            <p className="wardrobe-status">Loading wardrobe items...</p>
          )}
          {wardrobeError && (
            <p className="wardrobe-status wardrobe-error">{wardrobeError}</p>
          )}
          {!wardrobeLoading && !wardrobeError && wardrobeItems.length === 0 && (
            <p className="wardrobe-status">No items in your wardrobe yet.</p>
          )}

          {wardrobeItems.length > 0 && (
            <>
              <div className="wardrobe-controls">
                <div className="wardrobe-control-group">
                  <span className="wardrobe-control-label">Sort:</span>
                  <button
                    className={`wardrobe-control-button ${sortOrder === "newest" ? "active" : ""}`}
                    onClick={() => setSortOrder("newest")}
                  >
                    Newest
                  </button>
                  <button
                    className={`wardrobe-control-button ${sortOrder === "oldest" ? "active" : ""}`}
                    onClick={() => setSortOrder("oldest")}
                  >
                    Oldest
                  </button>
                </div>
                <div className="wardrobe-control-group">
                  <span className="wardrobe-control-label">Filter:</span>
                  <select
                    className="wardrobe-select"
                    value={typeFilter}
                    onChange={(e) => setTypeFilter(e.target.value)}
                  >
                    <option value="all">All types</option>
                    {Array.from(
                      new Set(
                        wardrobeItems
                          .map((item) => item.tags?.garment_type)
                          .filter((t): t is string => Boolean(t))
                      )
                    ).map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {(() => {
                const filtered = wardrobeItems.filter((item) => {
                  if (typeFilter === "all") return true;
                  return (item.tags?.garment_type || "").toLowerCase() === typeFilter.toLowerCase();
                });

                const sorted = [...filtered].sort((a, b) => {
                  const aDate = a.created_at ? new Date(a.created_at).getTime() : 0;
                  const bDate = b.created_at ? new Date(b.created_at).getTime() : 0;
                  return sortOrder === "newest" ? bDate - aDate : aDate - bDate;
                });

                if (sorted.length === 0) {
                  return (
                    <p className="wardrobe-status">
                      No items match the selected filter.
                    </p>
                  );
                }

                return (
                  <div className="wardrobe-grid">
                    {sorted.map((item) => (
                      <div key={item._id} className="wardrobe-card">
                        <div className="wardrobe-image-wrapper">
                          {item.url ? (
                            <>
                              <img
                                src={item.url}
                                alt={item.tags?.garment_type || "Wardrobe item"}
                                className="wardrobe-image"
                              />
                              <a
                                href={item.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="wardrobe-open-link"
                              >
                                Open in new tab
                              </a>
                            </>
                          ) : (
                            <div className="wardrobe-placeholder">
                              No image URL
                            </div>
                          )}
                        </div>
                        <div className="wardrobe-meta">
                          <div className="wardrobe-meta-primary">
                            <span className="wardrobe-tag">
                              {item.tags?.garment_type || "Unknown type"}
                            </span>
                          </div>
                          <div className="wardrobe-meta-secondary">
                            {item.tags?.primary_color && (
                              <span className="wardrobe-chip">
                                {item.tags.primary_color}
                              </span>
                            )}
                            {item.tags?.season && (
                              <span className="wardrobe-chip">
                                {item.tags.season}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;

