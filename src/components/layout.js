import React from "react"
import { Link } from "gatsby"
import { ThemeToggler } from 'gatsby-plugin-dark-mode'
import { rhythm, scale } from "../utils/typography"
import "../global.css"
class Layout extends React.Component {
  render() {
    const { location, title, children } = this.props
    const rootPath = `${__PATH_PREFIX__}/`
    let header

    if (location.pathname === rootPath) {
      header = (
        <div className="header">
          <h1
            style={{
              ...scale(1.5),
              marginBottom: 0,
              marginTop: 0,
            }}
          >
            <Link
              style={{
                boxShadow: `none`,
                textDecoration: `none`,
                color: `inherit`,
              }}
              to={`/`}
            >
              {title}
            </Link>
          </h1>
          <ThemeToggler>
            {({ theme, toggleTheme }) => (
              <label className="switch">
                <input
                  type="checkbox"
                  onChange={e => toggleTheme(e.target.checked ? 'dark' : 'light')}
                  checked={theme === 'dark'}
                />
                  <span class="slider round"></span>
              </label>
            )}
          </ThemeToggler>
        </div>
      )
    } else {
      header = (
        <div className="header">
          <h3
            style={{
              fontFamily: "'Permanent Marker', cursive",
              // marginTop: "1rem",
              margin:"0 0 0 0"
            }}
          >
            <Link
              style={{
                boxShadow: `none`,
                textDecoration: `none`,
                color: `inherit`,
              }}
              to={`/`}
            >
              {title}
            </Link>
          </h3>
          <ThemeToggler>
            {({ theme, toggleTheme }) => (
              <label className="switch">
                <input
                  type="checkbox"

                  onChange={e => toggleTheme(e.target.checked ? 'dark' : 'light')}
                  checked={theme === 'dark'}
                />
                <span className="slider round"></span>
              </label>
            )}
          </ThemeToggler>
        </div>
      )
    }
    return (
      <div
        style={{
          marginLeft: `auto`,
          backgroundColor: 'var(--bg)',
          color: 'var(--textNormal)',
          // transition: 'color 0.2s ease-out, background 0.2s ease-out',
          marginRight: `auto`,
          maxWidth: rhythm(24),
          padding: `${rhythm(1.5)} ${rhythm(3 / 4)}`,
        }}
      >
        <header>{header}</header>
        <main>{children}</main>
        <footer>
          © {new Date().getFullYear()} Sharan Sajiv Menon
        </footer>
      </div>
    )
  }
}

export default Layout;
