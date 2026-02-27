#[inline(always)]
pub(crate) fn brent_minimize<F>(
    mut f: F,
    low: f64,
    high: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    let mut a = low;
    let mut c = high;
    if !(a < c) {
        std::mem::swap(&mut a, &mut c);
    }

    let eps = f64::EPSILON;
    let tol = tol.abs().max(1e-12);

    let mut x = 0.5 * (a + c);
    let mut w = x;
    let mut v = x;

    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;

    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for _ in 0..max_iter {
        let m = 0.5 * (a + c);
        let tol1 = tol * x.abs() + eps;
        let tol2 = 2.0 * tol1;

        if (x - m).abs() <= tol2 - 0.5 * (c - a) {
            break;
        }

        let mut u: f64;
        let use_parabolic = if e.abs() > tol1 {
            let mut p = (x - v) * ((x - w) * (fx - fv)) - (x - w) * ((x - v) * (fx - fw));
            let mut q = 2.0 * (((x - v) * (fx - fw)) - ((x - w) * (fx - fv)));

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            let mut ok = false;
            if q.abs() > eps {
                let sstep = p / q;
                u = x + sstep;

                if (u - a) >= tol2 && (c - u) >= tol2 && sstep.abs() < 0.5 * e.abs() {
                    ok = true;
                }
            }

            if ok {
                d = p / q;
                u = x + d;
                if (u - a) < tol2 || (c - u) < tol2 {
                    d = if x < m { tol1 } else { -tol1 };
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if !use_parabolic {
            e = if x < m { c - x } else { a - x };
            d = 0.3819660_f64 * e;
        }

        if d.abs() < tol1 {
            d = if d >= 0.0 { tol1 } else { -tol1 };
        }

        u = x + d;
        let fu = f(u);

        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                c = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u >= x {
                c = u;
            } else {
                a = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    (x, fx)
}
